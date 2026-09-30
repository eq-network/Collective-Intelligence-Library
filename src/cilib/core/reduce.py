"""
Streaming reduction over a run — folding metrics INTO the scan instead of after it.

``run_scan`` stacks ``trace_fn(state)`` into arrays with a leading time axis, and
every metric in this library then scores that stacked trace. That decomposition
is an *unfused* ``unfold ; fold``: it materializes O(T x trace) bytes so a fold
can walk them once and return, usually, a single float. The cost is not
incidental — it is the binding constraint on long or large runs:

    value_contagion, N=5000, 16 seeds, T=2000
        sparse `friendship` in the state ......   0.45 MB
        stacked trace .........................   1.28 GB     (~2800x the state)

    io_economy traces `technical`, an (N, N) matrix, EVERY step -> O(T * N^2),
        while its `spectral_margin` metric reads only the final slice.

A ``Reducer`` is the fused form: an accumulator living in the scan carry, so the
memory cost is O(accumulator) rather than O(T x trace). Same arithmetic, same
answer, no time axis retained.

    Reducer ::= { init     : (state0, T)        -> acc
                  update   : (acc, state, t, T) -> acc
                  finalize : (acc, T)           -> value }

``T`` (the step count) is threaded through all three because it is a static
Python int at trace time and window reducers need it — ``late_mean`` has to know
where the last quarter of the run begins. ``t`` is the traced scan index, so
``update`` must be branch-free in it (use ``jnp.where``, not ``if``).

**This does not replace tracing.** Reducing over the AGENT axis is what saves
memory; the time axis for a scalar is 8 KB at T=2000. So keep emitting cheap
(T,) scalar series through ``trace_fn`` — timelines, onset detection and the
window regressions all need them — and use reducers for anything whose natural
readout is per-agent or whose window is long. ``run_scan_reduce`` returns both.

Replay note: the scan tier is pure and seeded, so a rollout is bit-reproducible
from its key. A metric you did not declare up front is recoverable *exactly* by
re-running with a different reducer — it just costs another run. That is why
reducing hard is safe here: nothing has to be recorded "just in case".
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax, random

from .graph import GraphState
from .scan import RoundFn, TraceFn


@dataclasses.dataclass(frozen=True)
class Reducer:
    """A fold over the states of one run, accumulated in the scan carry.

    A reducer folds over exactly the ``T`` states that ``trace_fn`` would record
    — the states AFTER each round — so that its result matches the trace-based
    metric it replaces. ``init`` therefore has to return an **identity element**
    (zero for a sum, ``-inf`` for a max), never ``readout(initial_state)``:
    seeding with a real value folds over ``T + 1`` states and quietly includes
    the pre-run state the trace never sees.

    Args:
        init: ``(initial_state, n_steps) -> acc``. Builds the identity
            accumulator; the state is passed for its SHAPES and dtypes, not its
            values. The accumulator is part of the scan carry, so it must be
            shape-stable like any other carry.
        update: ``(acc, state, t, n_steps) -> acc``, called with the state AFTER
            each round. ``t`` is traced: no Python branching on it.
        finalize: ``(acc, n_steps) -> value``, called once on the final
            accumulator. This is where a running sum becomes a mean, or an
            accumulated per-agent vector becomes a Gini coefficient.
    """
    init: Callable[[GraphState, int], Any]
    update: Callable[[Any, GraphState, jnp.ndarray, int], Any]
    finalize: Callable[[Any, int], Any]


def run_scan_reduce(
    round_fn: RoundFn,
    init_state: GraphState,
    n_steps: int,
    key: Any,
    reducers: Dict[str, Reducer],
    trace_fn: Optional[TraceFn] = None,
) -> Tuple[GraphState, Any, Dict[str, Any]]:
    """``run_scan``, plus reducers folded in the carry. One compiled ``lax.scan``.

    Deliberately a separate function rather than a ``reducers=`` kwarg on
    ``run_scan``: this returns a 3-tuple, and ``run_scan``'s 2-tuple contract is
    load-bearing for every environment and experiment in the repo.

    Args:
        round_fn: Pure ``(state, t, key_t) -> state``, as in ``run_scan``.
        init_state: Initial ``GraphState``.
        n_steps: Static step count — also handed to every reducer.
        key: A PRNG key.
        reducers: ``{name: Reducer}``. Empty dict is legal (then this is just
            ``run_scan`` with an extra empty dict).
        trace_fn: Optional per-step readout, stacked as usual. Keep this to
            cheap (T,)-shaped scalars; per-agent fields are what reducers are
            for.

    Returns:
        ``(final_state, trace, reduced)``. ``trace`` is ``None`` when
        ``trace_fn`` is ``None``; ``reduced`` maps each name to its finalized
        value.
    """
    items = tuple(sorted(reducers.items()))       # deterministic carry ordering
    init_accs = tuple(r.init(init_state, n_steps) for _, r in items)
    final_state, final_accs, trace = run_scan_reduce_over(
        round_fn, init_state, init_accs, jnp.arange(n_steps), key, items, n_steps, trace_fn)
    reduced = {name: r.finalize(a, n_steps)
               for a, (name, r) in zip(final_accs, items)}
    return final_state, trace, reduced


def run_scan_reduce_over(round_fn, init_state, init_accs, ts, key, items, n_steps,
                         trace_fn=None):
    """The scan of :func:`run_scan_reduce` alone: accumulators already built, not yet
    finalized, step indices given. Returns ``(final_state, final_accs, trace)``.

    Split out so ``EnvSpec.run_reduced`` can compile just the scan and keep
    ``init`` / ``finalize`` eager, as ``core.scan.run_scan_over`` does for
    ``run``: that is what keeps the compiled path bit-identical to this one.
    ``items`` is the sorted ``((name, Reducer), ...)`` tuple.
    """
    def body(carry, t):
        state, k, accs = carry
        k, step_key = random.split(k)
        new_state = round_fn(state, t, step_key)
        accs = tuple(r.update(a, new_state, t, n_steps)
                     for a, (_, r) in zip(accs, items))
        out = trace_fn(new_state) if trace_fn is not None else None
        return (new_state, k, accs), out

    (final_state, _, final_accs), trace = lax.scan(
        body, (init_state, key, init_accs), ts)
    return final_state, final_accs, trace


def run_scan_reduce_batch(
    round_fn: RoundFn,
    init_fn: Callable[[Any], GraphState],
    n_steps: int,
    keys: Any,
    reducers: Dict[str, Reducer],
    trace_fn: Optional[TraceFn] = None,
) -> Tuple[GraphState, Any, Dict[str, Any]]:
    """``vmap`` :func:`run_scan_reduce` over seeds — the sweep path.

    This is where reducers earn their keep: the batch axis multiplies whatever
    the trace costs, so a sweep that cannot hold ``(B, T, N)`` can still hold
    ``(B,)`` per metric.

    Returns:
        ``(final_states, traces, reduced)``, each with a leading ``(B,)`` axis.
        ``reduced[name]`` is shaped ``(B,)`` for a scalar metric — one value per
        seed, ready for the bootstrap CIs the benchmark harness computes.
    """
    def one(k):
        k_init, k_run = random.split(k)
        return run_scan_reduce(round_fn, init_fn(k_init), n_steps, k_run,
                               reducers, trace_fn)

    return jax.vmap(one)(keys)
