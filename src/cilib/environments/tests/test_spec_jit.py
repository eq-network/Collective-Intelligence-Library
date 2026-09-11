"""``EnvSpec.run*`` compile once per program and match the bare scan bit for bit.

The perf baseline found that a bare ``lax.scan`` recompiles on every call; ``EnvSpec``
now builds each scan under ``jax.jit`` and caches it, with ``init_fn`` kept op by op
outside the program. Equivalence is tested against ``core.scan`` / ``core.reduce``
called directly, across every registered environment at its default configuration.
Exact equality, not ``allclose``: if compilation rounds differently that is a finding
to record, not a tolerance to hide.

Two findings from building this test, both recorded in ``spec.py``'s docstring:
``init_fn`` under ``jit`` drifts in most environments (so it stays eager), and the
old ``run_batch`` passed seed-constant init leaves into the batched scan unbatched,
which changed the accumulation order in five environments. The batch reference here
is therefore the fully batched form (every leaf carries the seed axis), which is
what ``run_batch`` now computes and what per-seed ``run`` calls approximate best.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from cilib.core.reduce import run_scan_reduce
from cilib.core.scan import run_scan
from cilib.environments import REGISTRY, make_env
from cilib.environments.value_contagion.metrics import make_reducers

T = 12
B = 3


def _identical(a, b, where=""):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), f"{where}: leaf count {len(la)} != {len(lb)}"
    for x, y in zip(la, lb):
        x, y = jnp.asarray(x), jnp.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype, \
            f"{where}: {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}"
        if not bool(jnp.array_equal(x, y, equal_nan=True)):
            diff = jnp.abs(x - y)
            raise AssertionError(
                f"{where}: max |diff| = {float(diff.max())} over {int((diff > 0).sum())} elements")


def _split_batch(key, n):
    k_init, k_run = jax.vmap(jr.split, out_axes=1)(jr.split(key, n))
    return k_init, k_run


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_run_matches_bare_scan(name):
    env = make_env(name)
    key = jr.PRNGKey(0)
    k_init, k_run = jr.split(key)
    eager = run_scan(env.round_fn, env.init_fn(k_init), T, k_run, trace_fn=env.trace_fn)
    compiled = env.run(key, T)
    _identical(eager, compiled, f"{name} run")
    assert eager[0].global_attrs.keys() == compiled[0].global_attrs.keys()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_run_batch_matches_bare_batched_scan(name):
    env = make_env(name)
    key = jr.PRNGKey(1)
    k_init, k_run = _split_batch(key, B)
    states = jax.vmap(env.init_fn)(k_init)
    eager = jax.vmap(lambda s, k: run_scan(env.round_fn, s, T, k, trace_fn=env.trace_fn))(
        states, k_run)
    _identical(eager, env.run_batch(key, B, T), f"{name} run_batch")


def test_run_reduced_matches_bare_reduce():
    env = make_env("value_contagion")
    reducers = make_reducers(env.config)
    key = jr.PRNGKey(2)
    k_init, k_run = jr.split(key)
    eager = run_scan_reduce(env.round_fn, env.init_fn(k_init), T, k_run, reducers, trace_fn=None)
    _identical(eager, env.run_reduced(key, T, reducers), "value_contagion run_reduced")


def test_run_reduced_batch_matches_bare_batched_reduce():
    env = make_env("value_contagion")
    reducers = make_reducers(env.config)
    key = jr.PRNGKey(7)
    k_init, k_run = _split_batch(key, B)
    states = jax.vmap(env.init_fn)(k_init)
    eager = jax.vmap(lambda s, k: run_scan_reduce(env.round_fn, s, T, k, reducers, trace_fn=None))(
        states, k_run)
    _identical(eager, env.run_reduced_batch(key, B, T, reducers), "value_contagion run_reduced_batch")


def test_program_is_reused_for_the_same_shape():
    env = make_env("governed_commons")
    key = jr.PRNGKey(3)
    env.run(key, T)
    prog_a = env._program("scan", T, env.trace_fn)
    env.run(jr.PRNGKey(4), T)
    prog_b = env._program("scan", T, env.trace_fn)
    assert prog_a is prog_b, "same (kind, n_steps, trace_fn) must reuse the compiled program"
    assert len(env._programs) == 1
    env.run(key, T + 1)
    assert len(env._programs) == 2, "a different n_steps is a different program"
    env.run_batch(key, 2, T)
    assert len(env._programs) == 3, "a batch is a different program"
    env.run_batch(key, 5, T)
    assert len(env._programs) == 3, "the seed count is a shape of the input, not of the program"


def test_fresh_trace_fn_is_a_fresh_program():
    env = make_env("governed_commons")
    key = jr.PRNGKey(5)
    env.run(key, T, trace_fn=lambda s: s.global_attrs["step"])
    env.run(key, T, trace_fn=lambda s: s.global_attrs["step"])
    assert len(env._programs) == 2


def test_run_and_one_seed_of_batch_agree():
    # The documented key convention survives compilation: seed i of a batch is run(keys[i]).
    # Exact here; environments whose batched kernels accumulate in a different order than
    # the single-seed ones agree only to float32 rounding, which is a property of vmap, not
    # of compilation (see the module docstring).
    env = make_env("governed_commons")
    key = jr.PRNGKey(6)
    keys = jr.split(key, 2)
    finals, traces = env.run_batch(key, 2, T)
    final_1, trace_1 = env.run(keys[1], T)
    _identical(jax.tree_util.tree_map(lambda x: x[1], finals), final_1, "seed 1 final")
    _identical(jax.tree_util.tree_map(lambda x: x[1], traces), trace_1, "seed 1 trace")
