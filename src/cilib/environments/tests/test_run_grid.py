"""``EnvSpec.run_grid``: dial values are runtime data, one compile per grid shape.

Invariants (docs/run-grid-design.md):
1. Lifting a dial from closed-over config into ``global_attrs`` changes no number
   at its default value, on every existing run path.
2. Varying grid values adds no program; a new cell count adds exactly one.
3. With ``batch=1`` a cell is bit-identical to ``run_reduced_batch`` at the same
   value, except a reducer ``init`` the scan never overwrites (one ulp at most);
   with ``batch > 1`` it matches within float32 rounding.
4. A grid key must already be an array in ``global_attrs``; nothing else is accepted.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from cilib.core.category import transform
from cilib.core.reduce import Reducer
from cilib.core.graph import GraphState
from cilib.core.schedule import scheduled
from cilib.environments import make_env
from cilib.environments.spec import EnvSpec, with_globals
from cilib.metrics.reducers import at_step, late_mean, last, running_var

from .test_spec_jit import _assert_bit_identical

T, S, N = 24, 3, 64

# For batch > 1 only: float32 rounding on these short toy runs. Long runs of a
# thresholded model can drift much further (docs/run-grid-design.md, "Tolerance").
RTOL, ATOL = 1e-5, 1e-6


@transform(reads=["x"], writes=["x"])
def _push(state):
    return state.update_node_attrs("x", state.node_attrs["x"] + 1.0)


def _toy(rate=0.9, onset=5, lifted=True):
    """x <- rate * x + tanh(x) + noise, plus a +1 push switching on at ``onset``.

    ``lifted=False`` is the same model with both dials closed over, the form a
    dial has before it is promoted to a ``*_now`` global."""
    def init_fn(k):
        g = {"step": jnp.array(0, jnp.int32), "label": 1.0}
        if lifted:
            g.update(rate_now=jnp.array(rate, jnp.float32),
                     t_on_now=jnp.array(onset, jnp.int32))
        return GraphState(node_types=jnp.zeros(N, jnp.int32),
                          node_attrs={"x": jr.normal(k, (N,))}, adj_matrices={},
                          global_attrs=g)

    push = scheduled(_push, onset="t_on_now" if lifted else onset)

    def round_fn(s, t, k):
        r = s.global_attrs["rate_now"] if lifted else rate
        x = s.node_attrs["x"]
        s = s.update_node_attrs("x", r * x + 0.1 * jnp.tanh(x) + 0.05 * jr.normal(k, (N,)))
        s = push(s)
        return s.update_global_attr("step", s.global_attrs["step"] + 1)

    return EnvSpec(name="toy", config=None, init_fn=init_fn, round_fn=round_fn,
                   trace_fn=lambda s: {"m": jnp.mean(s.node_attrs["x"])})


def _mean_x(s):
    return jnp.mean(s.node_attrs["x"])


REDUCERS = {"late": late_mean(_mean_x), "last": last(_mean_x), "var": running_var(_mean_x),
            "at_on": at_step(_mean_x, "t_on_now")}
CLOSED_REDUCERS = {k: v for k, v in REDUCERS.items() if k != "at_on"}


def _close(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(np.asarray(x, float), np.asarray(y, float),
                                   rtol=RTOL, atol=ATOL)


# --- invariant 1: lifting a dial changes nothing at its default --------------------

def test_lifting_a_dial_is_bit_exact_on_every_path():
    lifted, closed = _toy(), _toy(lifted=False)
    key = jr.PRNGKey(1)
    # final states differ by the two extra globals, so compare the traces
    _assert_bit_identical(lifted.run(key, T)[1], closed.run(key, T)[1])
    _assert_bit_identical(lifted.run_batch(key, S, T)[1], closed.run_batch(key, S, T)[1])
    _assert_bit_identical(lifted.run_reduced_batch(key, S, T, CLOSED_REDUCERS)[1:],
                          closed.run_reduced_batch(key, S, T, CLOSED_REDUCERS)[1:])


def test_string_onset_equals_int_onset():
    lifted, closed = _toy(onset=7), _toy(onset=7, lifted=False)
    key = jr.PRNGKey(2)
    _assert_bit_identical(lifted.run_batch(key, S, T)[1], closed.run_batch(key, S, T)[1])


# --- invariant 2: one program per grid shape ---------------------------------------

def test_new_values_reuse_the_program_and_a_new_shape_adds_one():
    env = _toy()
    grid = lambda r: {"rate_now": jnp.array(r), "t_on_now": jnp.array([3, 9])}
    env.run_grid(jr.PRNGKey(0), S, T, grid([0.5, 0.9]), REDUCERS)
    n = len(env._programs)
    env.run_grid(jr.PRNGKey(4), S, T, grid([0.1, -0.3]), REDUCERS)
    assert len(env._programs) == n
    assert all(p._cache_size() == 1 for p, *_ in env._programs.values())
    for i, call in enumerate([
            lambda: env.run_grid(jr.PRNGKey(0), S, T, {"rate_now": jnp.array([0.5, 0.9, 0.7]),
                                                      "t_on_now": jnp.array([3, 9, 1])},
                                 REDUCERS),
            lambda: env.run_grid(jr.PRNGKey(0), S + 1, T, grid([0.5, 0.9]), REDUCERS),
            lambda: env.run_grid(jr.PRNGKey(0), S, T + 1, grid([0.5, 0.9]), REDUCERS),
            lambda: env.run_grid(jr.PRNGKey(0), S, T, grid([0.5, 0.9]), CLOSED_REDUCERS),
            lambda: env.run_grid(jr.PRNGKey(0), S, T, grid([0.5, 0.9]), REDUCERS, batch=2)]):
        call()
        assert len(env._programs) == n + i + 1


def test_batch_is_part_of_the_program():
    """A vectorized program must never answer a batch=1 call: only batch=1 is exact."""
    env, key = _toy(), jr.PRNGKey(3)
    grid = {"rate_now": jnp.array([0.5, 0.9, 0.7])}
    env.run_grid(key, S, T, grid, CLOSED_REDUCERS, batch=3)
    want = _toy(0.9).run_reduced_batch(key, S, T, CLOSED_REDUCERS)
    _assert_bit_identical(_cell(env.run_grid(key, S, T, grid, CLOSED_REDUCERS), 1), want)


# --- invariant 3: a cell matches the batch path ------------------------------------

def _cell(out, g):
    return jax.tree_util.tree_map(lambda x: x[g], out)


@pytest.mark.parametrize("rate,onset", [(0.9, 5), (0.6, 0), (1.02, 15)])
def test_cell_matches_run_reduced_batch_at_the_same_value(rate, onset):
    key = jr.PRNGKey(3)
    want = _toy(rate, onset).run_reduced_batch(key, S, T, CLOSED_REDUCERS)
    grid = {"rate_now": jnp.array([0.5, rate, 0.7]), "t_on_now": jnp.array([2, onset, 20])}
    _assert_bit_identical(_cell(_toy().run_grid(key, S, T, grid, CLOSED_REDUCERS), 1), want)
    _close(_cell(_toy().run_grid(key, S, T, grid, CLOSED_REDUCERS, batch=3), 1), want)


def test_at_step_is_exact_where_the_scan_writes_it():
    """Interior onsets are written by the scan, so they are exact; onset 0 and onsets
    past the run keep the init value, built inside the program (one ulp at most)."""
    key, onsets = jr.PRNGKey(5), (0, 1, 7, T, T + 3)
    _, _, got = _toy().run_grid(key, S, T, {"t_on_now": jnp.array(onsets)}, REDUCERS)
    for g, on in enumerate(onsets):
        _, _, want = _toy(onset=on).run_reduced_batch(key, S, T, REDUCERS)
        if 0 < on <= T:
            _assert_bit_identical(got["at_on"][g], want["at_on"])
        else:
            np.testing.assert_allclose(got["at_on"][g], want["at_on"], rtol=2e-7, atol=0)


def test_onset_grid_equals_separate_int_onset_runs():
    key, onsets = jr.PRNGKey(5), (0, 4, 11, T + 3)
    _, _, got = _toy().run_grid(key, S, T, {"t_on_now": jnp.array(onsets)}, CLOSED_REDUCERS)
    for g, on in enumerate(onsets):
        _, _, want = _toy(onset=on, lifted=False).run_reduced_batch(key, S, T, CLOSED_REDUCERS)
        _assert_bit_identical({k: v[g] for k, v in got.items()}, want)


@pytest.mark.parametrize("dial,field", [("churn_now", "churn"),
                                        ("repair_rate_now", "repair_rate")])
def test_cells_match_make_env_cells_on_ledger_society(dial, field):
    """A real model with lifted dials: each cell against a spec built with its value."""
    reducers = {"inf": late_mean(lambda s: jnp.sum(s.node_attrs["influence"]
                                                   * (s.node_types == 0)))}
    key, values, t = jr.PRNGKey(6), (0.0, 0.01, 0.2), 100
    got = make_env("ledger_society").run_grid(key, S, t, {dial: jnp.array(values)}, reducers)
    for g, v in enumerate(values):
        want = make_env("ledger_society", **{field: v}).run_reduced_batch(key, S, t, reducers)
        _assert_bit_identical(_cell(got, g), want)


def test_cells_share_the_seed_keys():
    """Common random numbers: equal dial values in two cells give equal numbers."""
    env = _toy()
    _, _, red = env.run_grid(jr.PRNGKey(7), S, T, {"rate_now": jnp.array([0.8, 0.5, 0.8])},
                             REDUCERS)
    _assert_bit_identical({k: v[0] for k, v in red.items()}, {k: v[2] for k, v in red.items()})


# --- chunking, tracing callers, at_step --------------------------------------------

@pytest.mark.parametrize("batch", [2, 5, 8])       # a remainder, all cells, more than G
def test_batched_cells_match_sequential_cells(batch):
    env, key = _toy(), jr.PRNGKey(8)
    grid = {"rate_now": jnp.linspace(0.3, 1.0, 5), "t_on_now": jnp.arange(5) * 3}
    got = env.run_grid(key, S, T, grid, REDUCERS, batch=batch)
    assert jax.tree_util.tree_leaves(got)[0].shape[:2] == (5, S)
    _close(got, env.run_grid(key, S, T, grid, REDUCERS))


def test_inside_a_caller_jit():
    env, key = _toy(), jr.PRNGKey(9)
    rates = jnp.array([0.4, 0.9])
    want = env.run_grid(key, S, T, {"rate_now": rates}, REDUCERS)
    got = jax.jit(lambda r: env.run_grid(key, S, T, {"rate_now": r}, REDUCERS))(rates)
    _close(got, want)
    assert len(env._programs) == 1          # the caller's jit compiled the second call


def test_at_step_reads_the_state_before_onset():
    env, key = _toy(), jr.PRNGKey(10)
    onsets = jnp.array([0, 1, 6, T])
    _, traces, red = env.run_grid(key, S, T, {"t_on_now": onsets}, REDUCERS,
                                  trace_fn=env.trace_fn)
    s0_mean = jax.vmap(lambda k: _mean_x(env.init_fn(jr.split(k)[0])))(jr.split(key, S))
    np.testing.assert_allclose(red["at_on"][0], s0_mean, rtol=RTOL)   # init, compiled
    for g, on in enumerate([1, 6, T]):
        np.testing.assert_array_equal(red["at_on"][g + 1], traces["m"][g + 1, :, on - 1])


# --- invariant 4: only array globals are dials -------------------------------------

@pytest.mark.parametrize("grid,match", [
    ({"rate": jnp.array([0.5])}, "not a jax.Array"),            # absent: closed over
    ({"label": jnp.array([2.0])}, "not a jax.Array"),           # Python scalar: static aux
    ({"rate_now": jnp.array(0.5)}, "expected"),                 # no cell axis
    ({"rate_now": jnp.ones((2, 2))}, "expected"),               # wrong entry shape
    ({"rate_now": jnp.ones(2), "t_on_now": jnp.ones(3)}, "common"),
    ({}, "at least one"),
])
def test_rejects_grids_that_are_not_runtime_dials(grid, match):
    with pytest.raises(ValueError, match=match):
        _toy().run_grid(jr.PRNGKey(0), S, T, grid, REDUCERS)


def test_rejects_weakly_typed_entries():
    env = _toy()

    def init_fn(k):
        s = env.init_fn(k)
        return s.replace(global_attrs={**s.global_attrs, "rate_now": jnp.asarray(0.9)})
    weak = EnvSpec("weak", None, init_fn, env.round_fn)
    with pytest.raises(ValueError, match="weakly typed"):
        weak.run_grid(jr.PRNGKey(0), S, T, {"rate_now": jnp.array([0.5])}, REDUCERS)


def test_grid_values_take_the_entry_dtype():
    finals, _, _ = _toy().run_grid(jr.PRNGKey(0), S, T, {"t_on_now": [2.0, 5.0]}, REDUCERS)
    assert finals.global_attrs["t_on_now"].dtype == jnp.int32
    np.testing.assert_array_equal(finals.global_attrs["t_on_now"][:, 0], [2, 5])


@pytest.mark.parametrize("key,values,match", [
    ("t_on_now", [2.9, 2.0], "not int32 integers"),
    ("step", [0, 10], "cannot be grid keys"),
    ("rng_key", [[0, 1], [2, 3]], "cannot be grid keys"),
])
def test_rejects_lossy_or_clock_keys(key, values, match):
    with pytest.raises(ValueError, match=match):
        _toy().run_grid(jr.PRNGKey(0), S, T, {key: values}, REDUCERS)


def test_reducer_init_sees_the_cell_value():
    probe = Reducer(init=lambda s, T: s.global_attrs["rate_now"],
                    update=lambda acc, s, t, T: acc, finalize=lambda acc, T: acc)
    rates = [0.25, 0.75]
    _, _, red = _toy().run_grid(jr.PRNGKey(0), S, T, {"rate_now": rates}, {"r": probe})
    np.testing.assert_array_equal(red["r"], np.repeat(np.float32(rates)[:, None], S, 1))


def test_with_globals_replaces_only_array_entries():
    s = _toy().init_fn(jr.PRNGKey(0))
    out = with_globals(s, {"rate_now": jnp.array(0.1, jnp.float32)})
    assert float(out.global_attrs["rate_now"]) == pytest.approx(0.1)
    assert float(s.global_attrs["rate_now"]) == pytest.approx(0.9)       # input untouched
    for bad in ("label", "missing"):
        with pytest.raises(KeyError):
            with_globals(s, {bad: jnp.array(1.0)})
