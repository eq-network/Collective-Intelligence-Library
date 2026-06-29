"""
Tests for the pure compiled scan tier (core/scan.py) and the scan-safe
conditional transform (core/category.gated).

These validate the four claims the time-model fix rests on:
  1. run_scan matches an eager Python reference loop (correctness).
  2. The round body is TRACED, not Python-looped (it compiles; the body runs a
     constant number of times regardless of n_steps).
  3. PRNG is threaded deterministically (same key -> same trajectory).
  4. run_scan_batch vmaps over seeds as one program and matches per-seed runs.
  5. gated() fires only when its predicate holds (e.g. after t_shift), inside scan.

Run: python -m pytest tests/test_scan.py -q   (from repo root)
"""
import jax
import jax.numpy as jnp
from jax import random

from cilib.core.graph import GraphState
from cilib.core.scan import run_scan, run_scan_batch
from cilib.core.category import gated, sequential


def _tiny_state(n=3, val=0.0):
    """A minimal active (non-capacity) GraphState: n nodes, scalar attr 'x'."""
    return GraphState(
        node_types=jnp.zeros(n, dtype=jnp.int32),
        node_attrs={"x": jnp.full((n,), val, dtype=jnp.float32)},
        adj_matrices={"trust": jnp.eye(n, dtype=jnp.float32)},
        global_attrs={},
    )


def test_run_scan_matches_eager_reference():
    """Deterministic round (x += 1) — final state and trace match a Python loop."""
    T = 25
    init = _tiny_state()

    def round_fn(state, t, key):  # deterministic; ignores t, key
        return state.update_node_attrs("x", state.node_attrs["x"] + 1.0)

    final, trace = run_scan(round_fn, init, T, random.PRNGKey(0),
                            trace_fn=lambda s: s.node_attrs["x"])

    # Eager reference
    ref = init
    for _ in range(T):
        ref = round_fn(ref, 0, None)

    assert jnp.allclose(final.node_attrs["x"], ref.node_attrs["x"])
    assert jnp.allclose(final.node_attrs["x"], jnp.full(3, float(T)))
    # Trace stacks along a leading time axis of length T.
    assert trace.shape == (T, 3)
    assert jnp.allclose(trace[-1], jnp.full(3, float(T)))


def test_round_body_is_traced_not_python_looped():
    """The Python body runs a constant (tiny) number of times, not n_steps times."""
    calls = [0]

    def round_fn(state, t, key):
        calls[0] += 1  # increments once per *trace*, not once per step
        return state.update_node_attrs("x", state.node_attrs["x"] + 1.0)

    T = 200
    final, _ = run_scan(round_fn, _tiny_state(), T, random.PRNGKey(0))

    assert calls[0] <= 3, f"body Python-evaluated {calls[0]} times (expected <=3; compiled)"
    assert calls[0] != T
    assert jnp.allclose(final.node_attrs["x"], jnp.full(3, float(T)))


def test_prng_threading_is_deterministic():
    """Stochastic round: same key -> identical trajectory; different key -> different."""
    def round_fn(state, t, key):
        return state.update_node_attrs("x", state.node_attrs["x"] + random.normal(key, (3,)))

    a, _ = run_scan(round_fn, _tiny_state(), 30, random.PRNGKey(7))
    b, _ = run_scan(round_fn, _tiny_state(), 30, random.PRNGKey(7))
    c, _ = run_scan(round_fn, _tiny_state(), 30, random.PRNGKey(8))

    assert jnp.allclose(a.node_attrs["x"], b.node_attrs["x"])          # reproducible
    assert not jnp.allclose(a.node_attrs["x"], c.node_attrs["x"])      # key actually used
    # A pure +1 deterministic part would give exactly T; randomness must perturb it.
    assert not jnp.allclose(a.node_attrs["x"], jnp.zeros(3))


def test_run_scan_batch_matches_per_seed():
    """vmap over seeds equals looping run_scan per seed with the same key discipline."""
    T, B = 20, 8

    def init_fn(key):
        # per-seed init: small random offset so seeds genuinely differ
        return _tiny_state().update_node_attrs("x", random.normal(key, (3,)))

    def round_fn(state, t, key):
        return state.update_node_attrs("x", state.node_attrs["x"] + random.normal(key, (3,)))

    keys = random.split(random.PRNGKey(0), B)
    batch_final, batch_trace = run_scan_batch(round_fn, init_fn, T, keys,
                                              trace_fn=lambda s: s.node_attrs["x"])

    assert batch_final.node_attrs["x"].shape == (B, 3)
    assert batch_trace.shape == (B, T, 3)

    # Per-seed reference replicating run_scan_batch's split discipline.
    for i in range(B):
        k_init, k_run = random.split(keys[i])
        ref_final, _ = run_scan(round_fn, init_fn(k_init), T, k_run)
        assert jnp.allclose(batch_final.node_attrs["x"][i], ref_final.node_attrs["x"])


def test_gated_fires_only_after_tshift():
    """gated() inside a scanned round increments a counter only for t >= t_shift."""
    T, t_shift = 50, 30
    init = _tiny_state().update_node_attrs("fired", jnp.zeros(3, dtype=jnp.float32))

    def inc(state):  # a plain Transform
        return state.update_node_attrs("fired", state.node_attrs["fired"] + 1.0)

    def round_fn(state, t, key):
        # predicate closes over the scan index t — a time gate
        step = sequential(gated(lambda s: t >= t_shift, inc))
        return step(state)

    final, _ = run_scan(round_fn, init, T, random.PRNGKey(0))

    expected = float(T - t_shift)  # number of steps with t in [t_shift, T)
    assert jnp.allclose(final.node_attrs["fired"], jnp.full(3, expected)), \
        f"gated fired {float(final.node_attrs['fired'][0])} times, expected {expected}"


def test_gated_off_is_identity():
    """A gate whose predicate is always False leaves state untouched (baseline)."""
    T = 10
    init = _tiny_state(val=5.0)

    def inc(state):
        return state.update_node_attrs("x", state.node_attrs["x"] + 1.0)

    def round_fn(state, t, key):
        return gated(lambda s: jnp.bool_(False), inc)(state)

    final, _ = run_scan(round_fn, init, T, random.PRNGKey(0))
    assert jnp.allclose(final.node_attrs["x"], jnp.full(3, 5.0))


if __name__ == "__main__":
    test_run_scan_matches_eager_reference()
    test_round_body_is_traced_not_python_looped()
    test_prng_threading_is_deterministic()
    test_run_scan_batch_matches_per_seed()
    test_gated_fires_only_after_tshift()
    test_gated_off_is_identity()
    print("all scan tests passed")
