"""``EnvSpec.run`` / ``run_batch`` compile once per spec and change no number.

The compiled boundary is only worth having if every seed still produces the
trajectory it produced on the eager path; otherwise results already reported
from ``(env, config, seed)`` stop reproducing. So the claim is bit-identity,
not closeness, against ``core.scan``'s eager functions (the pre-change code),
for every registered environment and for the three constructions where a
plain ``jax.jit`` was measured to drift: a jitted ``init_fn``, closed-over
config arrays folded as constants, and batching fields ``vmap`` left unbatched.
A tolerance here would hide exactly that drift.
"""
import copy

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from cilib.core.graph import GraphState
from cilib.core.scan import run_scan, run_scan_batch
from cilib.environments import list_envs, make_env
from cilib.environments.spec import EnvSpec

T, S = 20, 3

# Registered defaults, plus the one non-default representation the engine
# carries: a sparse BCOO adjacency, whose pytree aux holds its shape.
CASES = [(n, {}) for n in list_envs()] + [("value_contagion", {"sparse_friendship": True})]
IDS = [n + ("-sparse" if kw else "") for n, kw in CASES]


def _assert_bit_identical(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.array_equal(x, y, equal_nan=True)


def _eager_run(env, key, n_steps):
    k_init, k_run = jr.split(key)
    return run_scan(env.round_fn, env.init_fn(k_init), n_steps, k_run, trace_fn=env.trace_fn)


def _eager_batch(env, key, n_seeds, n_steps):
    return run_scan_batch(env.round_fn, env.init_fn, n_steps, jr.split(key, n_seeds),
                          trace_fn=env.trace_fn)


@pytest.mark.parametrize("name,kw", CASES, ids=IDS)
def test_run_matches_eager_bit_for_bit(name, kw):
    env = make_env(name, **kw)
    _assert_bit_identical(env.run(jr.PRNGKey(7), T), _eager_run(env, jr.PRNGKey(7), T))


@pytest.mark.parametrize("name,kw", CASES, ids=IDS)
def test_run_batch_matches_eager_bit_for_bit(name, kw):
    env = make_env(name, **kw)
    _assert_bit_identical(env.run_batch(jr.PRNGKey(7), S, T),
                          _eager_batch(env, jr.PRNGKey(7), S, T))


def _toy(init_extra=lambda: {}):
    """A model whose round applies transcendentals to a closed-over config array:
    under a plain jit XLA folds those as constants and the bits change."""
    n = 256
    rates = jnp.linspace(-4.0, 4.0, n, dtype=jnp.float32)

    def init_fn(k):
        return GraphState(node_types=jnp.zeros(n, jnp.int32),
                          node_attrs={"x": jr.normal(k, (n,)), **init_extra()},
                          adj_matrices={})

    def round_fn(s, t, k):
        drive = jnp.tanh(rates) + jnp.exp(-jnp.abs(rates)) + jnp.log1p(rates * rates)
        x = 0.5 * s.node_attrs["x"] + 0.1 * drive + 0.01 * jr.normal(k, (n,))
        return s.update_node_attrs("x", x * jnp.exp(-t / 10.0))

    return EnvSpec(name="toy", config=None, init_fn=init_fn, round_fn=round_fn,
                   trace_fn=lambda s: {"x": s.node_attrs["x"]})


def test_closed_over_config_arrays_are_not_folded():
    env = _toy()
    _assert_bit_identical(env.run(jr.PRNGKey(5), T), _eager_run(env, jr.PRNGKey(5), T))
    _assert_bit_identical(env.run_batch(jr.PRNGKey(5), S, T),
                          _eager_batch(env, jr.PRNGKey(5), S, T))


def test_custom_jvp_init_field_stays_unbatched():
    """``jax.nn.relu`` of a constant is an unbatched tracer under vmap; it must be
    scanned unbatched, as the eager path does."""
    env = _toy(lambda: {"w": jax.nn.relu(jnp.linspace(-1.0, 1.0, 256))})
    _assert_bit_identical(env.run_batch(jr.PRNGKey(2), S, T),
                          _eager_batch(env, jr.PRNGKey(2), S, T))


def test_repeat_calls_reuse_one_compile():
    """The point of the change: a second call with a new key but the same
    signature is a cache hit, not a new program."""
    env = make_env("value_contagion")
    for k in (0, 1):
        env.run(jr.PRNGKey(k), T)
        env.run_batch(jr.PRNGKey(k), S, T)
    assert len(env._programs) == 2
    assert all(p._cache_size() == 1 for p, *_ in env._programs.values())
    env.run(jr.PRNGKey(0), T + 1)                   # a new length is a new program
    assert len(env._programs) == 3


def test_inside_caller_transforms_the_old_program_is_used():
    """Callers wrap ``env.run`` / ``run_batch`` in their own jit or vmap
    (perf_baseline, lever tests). There the call must be the pre-change code
    inline, so the caller's compile sees the same program it saw before."""
    env = make_env("governed_commons")
    keys = jr.split(jr.PRNGKey(3), 2)
    _assert_bit_identical(jax.jit(jax.vmap(lambda k: env.run(k, T)))(keys),
                          jax.jit(jax.vmap(lambda k: _eager_run(env, k, T)))(keys))
    _assert_bit_identical(jax.jit(lambda k: env.run_batch(k, S, T))(keys[0]),
                          jax.jit(lambda k: _eager_batch(env, k, S, T))(keys[0]))


def test_inputs_the_eager_path_accepted():
    env = make_env("value_contagion")
    _assert_bit_identical(env.run(jr.PRNGKey(0), jnp.array(5)), env.run(jr.PRNGKey(0), 5))
    _, trace = env.run_batch(jr.PRNGKey(0), 0, 5)
    assert all(x.shape[0] == 0 for x in jax.tree_util.tree_leaves(trace))


def test_copies_do_not_share_compiled_programs():
    """A copy with a swapped ``round_fn`` must run its own round, not the
    original's cached program; and a spec that has run must still deepcopy."""
    env = make_env("value_contagion")
    env.run(jr.PRNGKey(0), T)
    clone = copy.copy(env)
    object.__setattr__(clone, "round_fn", lambda s, t, k: s)
    final, _ = clone.run(jr.PRNGKey(0), T)
    _assert_bit_identical(final, env.init_fn(jr.split(jr.PRNGKey(0))[0]))
    _assert_bit_identical(copy.deepcopy(env).run(jr.PRNGKey(1), T), env.run(jr.PRNGKey(1), T))
