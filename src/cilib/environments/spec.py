"""
The environment contract: ``EnvSpec``.

An environment in this library is not a class hierarchy — it is a bundle of pure functions over
``GraphState`` plus an evaluation suite. Formalizing that bundle as a single object is what makes
environments *reproducible by construction*: every environment that fills this contract inherits
the same compiled execution, ``vmap``-over-seeds batching, trajectory tracing, and metric wiring
for free, so adding a new one is "implement the five fields, register a name" rather than "wire a
bespoke run loop".

    EnvSpec ::= {
      config   : frozen params (closed over by the transforms, never stored in state)
      init_fn  : key -> GraphState                  (per-seed initial state; vmap-friendly)
      round_fn : (state, t, key) -> state           (one tick; == core.scan.RoundFn)
      trace_fn : state -> dict                       (per-step readout for trajectories)
      metrics  : { name : (trace -> scalar) }        (offline evaluation, e.g. the GovSim suite)
    }

``run`` / ``run_batch`` are thin wrappers over ``core.scan`` — the *only* place execution lives —
so the substrate code never touches the scan plumbing. Governance and measurement compose on top
by wrapping ``round_fn`` and extending ``trace_fn`` / ``metrics``; the contract does not change.

Compilation happens here, once per program. A bare ``lax.scan`` is compiled when it is called,
and a second bare call compiles again: the perf baseline (``experiments/perf_baseline``) measured
that repeat cost at 0.3 to 0.9 s against a 1 to 6 ms run. Each ``run*`` method therefore builds
its scan under ``jax.jit`` with the static arguments (``n_steps``, ``n_seeds``) closed over, and
keeps the compiled program on the spec, keyed by those statics plus the identity of the
``trace_fn`` and reducers it closes over. A second call with the same key reuses the executable.
Pass the same ``trace_fn`` object across calls to get that reuse; a fresh lambda per call is a
fresh program.

``init_fn`` deliberately runs op by op, outside the compiled program, and the initial state is
passed in as an argument. Measured 2026-09-11 across the registry at default size: the scan
compiles bit-identically to the bare path in every environment, but ``init_fn`` under ``jit``
rounds differently in 8 of 11 (XLA fuses the initial-state arithmetic; up to 2e-6 absolute on
float32 fields such as normalized preferences). Keeping init eager keeps ``run`` bit-identical
to what the same call produced before compilation was added, and init is a one-off cost per
call, not per step.

The batch path changed by one detail, found by the same measurement. The old ``run_batch``
traced ``init_fn`` and the scan under one ``vmap``, so any init leaf that did not depend on the
seed (a recipe matrix, a starting stock) entered the scan *unbatched* and was broadcast inside
it, and XLA accumulated those products in a different order than for a batched operand. In
five environments (the economy family and ``ledger_society``) that moved results at float32
rounding, and where a rounding difference crosses a discrete rule (a rewiring, an enforcement
threshold) it grows. ``run_batch`` now passes every leaf with its seed axis, which is what
per-seed ``run`` calls compute and what the batch was documented to equal. The old numbers are
reproduced exactly by unbatching those leaves (``in_axes=None``), so the change is understood,
not tolerated; batch results committed before 2026-09-11 for those five environments may
differ from a rerun at the last float32 digit.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.reduce import Reducer, run_scan_reduce
from cilib.core.scan import run_scan, RoundFn, TraceFn


# A metric here scores a *trajectory* (the stacked trace_fn output), not a single state — that is
# how the GovSim suite (survival_time, efficiency, ...) is shaped. Distinct from the inline
# per-step ``metrics/transform.make_metrics_transform``, which writes scalars into the scan carry.
MetricFn = Callable[[Any], Any]


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """A fully-specified, runnable environment. Construct via a registered builder (``make_env``)."""

    name: str
    config: Any
    init_fn: Callable[[Any], GraphState]
    round_fn: RoundFn
    trace_fn: Optional[TraceFn] = None
    metrics: Dict[str, MetricFn] = dataclasses.field(default_factory=dict)
    # Compiled programs, keyed by (kind, n_steps, batched, trace_fn, reducers). Excluded from
    # equality and repr; the dict is mutated in place, which a frozen dataclass permits.
    _programs: Dict[Any, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False)

    def run(self, key: Any, n_steps: int,
            trace_fn: Optional[TraceFn] = None) -> Tuple[GraphState, Any]:
        """One rollout. Returns ``(final_state, trace)``; trace fields have leading axis ``n_steps``.

        ``key`` is split into independent init / run keys (the ``run_scan_batch`` convention) so a
        single ``run`` and one seed of a ``run_batch`` produce the same trajectory for a given key.
        """
        tf = trace_fn if trace_fn is not None else self.trace_fn
        k_init, k_run = jr.split(key)
        return self._program("scan", n_steps, tf)(self.init_fn(k_init), k_run)

    def run_batch(self, key: Any, n_seeds: int, n_steps: int,
                  trace_fn: Optional[TraceFn] = None) -> Tuple[GraphState, Any]:
        """``vmap``ped multi-seed sweep. Returns ``(final_states, traces)`` with leading axis
        ``(n_seeds,)``; trace fields are shaped ``(n_seeds, n_steps, ...)``. Compiles to one
        ``vmap(scan(...))``."""
        tf = trace_fn if trace_fn is not None else self.trace_fn
        k_init, k_run = jax.vmap(jr.split, out_axes=1)(jr.split(key, n_seeds))
        return self._program("scan", n_steps, tf, batched=True)(
            jax.vmap(self.init_fn)(k_init), k_run)

    def run_reduced(self, key: Any, n_steps: int, reducers: Dict[str, Reducer],
                    trace_fn: Optional[TraceFn] = None
                    ) -> Tuple[GraphState, Any, Dict[str, Any]]:
        """One rollout scoring ``reducers`` INSIDE the scan — the constant-memory path.

        Use this when the stacked trace would not fit: ``metrics`` consume a
        materialized ``(T, N)`` trajectory, which is O(T) larger than the state
        it came from (see ``core/reduce.py`` for the measured ratio). Reducers
        fold in the carry instead, so cost is O(accumulator).

        ``trace_fn`` defaults to ``None`` here rather than to ``self.trace_fn``:
        the environments' default traces are exactly the per-agent arrays you are
        trying not to materialize, so inheriting one would defeat the call. Pass
        a deliberately cheap one — (T,)-shaped scalars are ~8 KB at T=2000 and
        worth keeping for timelines.

        Reducers are passed explicitly rather than carried on the spec because
        ``GameSpec`` is frozen (CLAUDE.md) and could not propagate them through
        ``close()``; an always-empty field would be dead weight. Environments
        expose a ``make_reducers(cfg)`` factory instead.

        Returns:
            ``(final_state, trace, reduced)``.
        """
        k_init, k_run = jr.split(key)
        return self._program("reduce", n_steps, trace_fn, reducers=reducers)(
            self.init_fn(k_init), k_run)

    def run_reduced_batch(self, key: Any, n_seeds: int, n_steps: int,
                          reducers: Dict[str, Reducer],
                          trace_fn: Optional[TraceFn] = None
                          ) -> Tuple[GraphState, Any, Dict[str, Any]]:
        """``vmap``ped sweep with in-scan reduction. ``reduced[name]`` is shaped
        ``(n_seeds,)`` for a scalar metric — already the per-seed vector the
        benchmark harness bootstraps over, with no trajectory kept."""
        k_init, k_run = jax.vmap(jr.split, out_axes=1)(jr.split(key, n_seeds))
        return self._program("reduce", n_steps, trace_fn, reducers=reducers, batched=True)(
            jax.vmap(self.init_fn)(k_init), k_run)

    def evaluate(self, trace: Any) -> Dict[str, Any]:
        """Score a trajectory (or batch of trajectories) with every metric in the suite."""
        return {name: fn(trace) for name, fn in self.metrics.items()}

    # -- compilation ---------------------------------------------------------------------------

    def _program(self, kind: str, n_steps: int, trace_fn: Optional[TraceFn],
                 reducers: Optional[Dict[str, Reducer]] = None,
                 batched: bool = False) -> Callable[[GraphState, Any], Any]:
        """The compiled ``(init_state, run_key) -> result`` program for one call shape.

        ``kind`` is ``"scan"`` (``run_scan``) or ``"reduce"`` (``run_scan_reduce``);
        ``batched`` wraps it in ``vmap`` over a leading seed axis of both arguments.
        The cache entry keeps a reference to the ``trace_fn`` and reducers it was
        keyed on, so their ``id``s cannot be recycled by a later object and alias
        a stale program.
        """
        items = tuple(sorted(reducers.items())) if reducers is not None else None
        ident = (kind, int(n_steps), batched, id(trace_fn),
                 None if items is None else tuple((n, id(r)) for n, r in items))
        hit = self._programs.get(ident)
        if hit is not None:
            return hit[0]

        round_fn = self.round_fn
        if kind == "scan":
            def one(state, key):
                return run_scan(round_fn, state, n_steps, key, trace_fn=trace_fn)
        elif kind == "reduce":
            reducers_d = dict(items)

            def one(state, key):
                return run_scan_reduce(round_fn, state, n_steps, key, reducers_d,
                                       trace_fn=trace_fn)
        else:
            raise ValueError(f"unknown program kind {kind!r}")

        compiled = jax.jit(jax.vmap(one) if batched else one)
        self._programs[ident] = (compiled, trace_fn, items)
        return compiled
