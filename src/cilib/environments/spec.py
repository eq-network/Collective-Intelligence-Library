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
so the substrate code never touches the scan plumbing. Both compile the scan once per spec and
call signature, so repeat calls skip the compile, and every seed's trajectory stays bit-identical
to the un-jitted path (``EnvSpec._compiled``, ``tests/test_spec_jit.py``). Governance and
measurement compose on top by wrapping ``round_fn`` and extending ``trace_fn`` / ``metrics``; the
contract does not change.
"""
from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.reduce import Reducer, run_scan_reduce, run_scan_reduce_batch
from cilib.core.scan import run_scan, run_scan_batch, run_scan_over, RoundFn, TraceFn


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

    def run(self, key: Any, n_steps: int,
            trace_fn: Optional[TraceFn] = None) -> Tuple[GraphState, Any]:
        """One rollout. Returns ``(final_state, trace)``; trace fields have leading axis ``n_steps``.

        ``key`` is split into independent init / run keys (the ``run_scan_batch`` convention) so a
        single ``run`` and one seed of a ``run_batch`` produce the same trajectory for a given key.
        """
        tf = trace_fn if trace_fn is not None else self.trace_fn
        n_steps = operator.index(n_steps)
        k_init, k_run = jr.split(key)
        state = self.init_fn(k_init)
        if _traced((state, k_run)):
            return run_scan(self.round_fn, state, n_steps, k_run, trace_fn=tf)

        def scan(state, k_run, ts):
            return run_scan_over(self.round_fn, state, ts, k_run, tf)
        return self._compiled("run", scan, (state, k_run, jnp.arange(n_steps)), tf)

    def run_batch(self, key: Any, n_seeds: int, n_steps: int,
                  trace_fn: Optional[TraceFn] = None) -> Tuple[GraphState, Any]:
        """``vmap``ped multi-seed sweep. Returns ``(final_states, traces)`` with leading axis
        ``(n_seeds,)``; trace fields are shaped ``(n_seeds, n_steps, ...)``. Compiles to one
        ``vmap(scan(...))``."""
        tf = trace_fn if trace_fn is not None else self.trace_fn
        n_steps = operator.index(n_steps)
        keys = jr.split(key, n_seeds)
        batch = _init_batch(self.init_fn, keys) if n_seeds and not _traced(keys) else None
        if batch is None:
            return run_scan_batch(self.round_fn, self.init_fn, n_steps, keys, trace_fn=tf)
        leaves, treedef, axes, k_run = batch

        def scan(leaves, k_run, ts):
            def one(ls, k):
                state = jax.tree_util.tree_unflatten(treedef, ls)
                return run_scan_over(self.round_fn, state, ts, k, tf)
            return jax.vmap(one, in_axes=(list(axes), 0))(leaves, k_run)
        return self._compiled("run_batch", scan, (leaves, k_run, jnp.arange(n_steps)), tf,
                              static=(treedef, axes))

    @functools.cached_property
    def _programs(self) -> dict:
        return {}

    def __getstate__(self):
        # copy / deepcopy / pickle start from an empty compile cache: a traced
        # program holds tracebacks that cannot be copied, and a shared cache would
        # hand a copy the original's program.
        return {k: v for k, v in self.__dict__.items() if k != "_programs"}

    def _compiled(self, tag, fn, args, trace_fn, static=()):
        """``fn(*args)`` as one XLA program, traced and compiled once per call signature.

        This is ``jax.jit(fn)`` with one difference that decides bit-identity with
        the eager scan it replaces: arrays ``fn`` closes over (config arrays
        captured by ``round_fn``) and the step indices reach XLA as runtime
        arguments, as they did in the eager scan, not as baked-in constants that
        XLA folds with the host's libm and rounds differently. ``init_fn`` stays
        outside for the same reason: compiled, its fused ops round differently
        (9 of the 13 registered environments).

        The cache lives on the spec, so a compile lives as long as its spec. It
        is keyed on ``tag``, the identity of ``round_fn`` and ``trace_fn`` (a
        fresh lambda per call is a fresh compile, as before), whatever ``fn``
        closes over that shapes the program (``static``), and the inputs' tree
        structure (which includes ``global_attrs``), shapes and dtypes.

        Not covered, because the eager path re-traced every call and this does
        not: a ``round_fn`` or ``trace_fn`` that reads mutable Python state (a
        numpy dial edited in place between calls) keeps the value it saw at the
        first call. Configs are frozen for this reason. Likewise a Python scalar
        in ``global_attrs`` is matched by ``==``, so ``-0.0`` reuses the program
        traced for ``0.0``; ``global_attrs`` must not vary per call.
        """
        flat, in_tree = jax.tree_util.tree_flatten(args)
        sig = (tag, id(trace_fn), id(self.round_fn), static, in_tree,
               tuple(jax.typeof(x) for x in flat))
        if sig not in self._programs:
            out_tree = []

            def flat_fn(*leaves):
                out = fn(*jax.tree_util.tree_unflatten(in_tree, leaves))
                out, tree = jax.tree_util.tree_flatten(out)
                out_tree.append(tree)
                return out
            closed = jax.make_jaxpr(flat_fn)(*flat)
            program = jax.jit(functools.partial(jax.core.eval_jaxpr, closed.jaxpr))
            # trace_fn and round_fn are held so their ids are not reused while cached.
            self._programs[sig] = (program, closed.consts, out_tree[0], (trace_fn, self.round_fn))
        program, consts, out_tree, _ = self._programs[sig]
        return jax.tree_util.tree_unflatten(out_tree, program(consts, *flat))

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
        return run_scan_reduce(self.round_fn, self.init_fn(k_init), n_steps,
                               k_run, reducers, trace_fn=trace_fn)

    def run_reduced_batch(self, key: Any, n_seeds: int, n_steps: int,
                          reducers: Dict[str, Reducer],
                          trace_fn: Optional[TraceFn] = None
                          ) -> Tuple[GraphState, Any, Dict[str, Any]]:
        """``vmap``ped sweep with in-scan reduction. ``reduced[name]`` is shaped
        ``(n_seeds,)`` for a scalar metric — already the per-seed vector the
        benchmark harness bootstraps over, with no trajectory kept."""
        keys = jr.split(key, n_seeds)
        return run_scan_reduce_batch(self.round_fn, self.init_fn, n_steps, keys,
                                     reducers, trace_fn=trace_fn)

    def evaluate(self, trace: Any) -> Dict[str, Any]:
        """Score a trajectory (or batch of trajectories) with every metric in the suite."""
        return {name: fn(trace) for name, fn in self.metrics.items()}


def _traced(tree) -> bool:
    """True inside a caller's ``jit`` / ``vmap`` / ``grad``: their trace compiles the whole
    call, so ``run`` / ``run_batch`` take the old inline path and add no program of their own."""
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(tree))


def _init_batch(init_fn, keys):
    """Per-seed initial states, built eagerly, in the layout ``run_scan_batch`` scans them.

    Under an eager ``vmap``, a state field that does not depend on the key (a fixed
    network, a zeroed ledger) is not batched, and the scan is compiled with that
    field unbatched. Batching it anyway compiles a different program with different
    float32 rounding, so each field's batching is recorded here and passed to the
    compiled scan as ``in_axes``. A field counts as batched when ``vmap`` says so
    (``batch_dim``), not merely when it is a tracer: ``custom_jvp`` functions such
    as ``jax.nn.relu`` return unbatched tracers.

    The treedef is taken per seed, inside the ``vmap``: a pytree with shape in its
    aux data (a sparse ``BCOO`` adjacency) comes out of ``vmap`` carrying the
    batched shape, and unflattening one seed's leaves with it is invalid.

    Returns ``(leaves, treedef, axes, run_keys)``, or ``None`` when an enclosing
    transformation traces the states (see ``_traced``).
    """
    seen = []

    def one(k):
        k_init, k_run = jr.split(k)
        state = init_fn(k_init)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        seen.append((treedef, tuple(getattr(x, "batch_dim", None) is not None for x in leaves)))
        return state, k_run

    states, k_run = jax.vmap(one)(keys)
    if _traced((states, k_run)):
        return None
    treedef, mapped = seen[0]
    leaves = [x if b else x[0] for x, b in zip(jax.tree_util.tree_leaves(states), mapped)]
    return leaves, treedef, tuple(0 if b else None for b in mapped), k_run
