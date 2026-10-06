"""
The environment contract: ``EnvSpec``.

An environment in this library is not a class hierarchy — it is a bundle of pure functions over
``GraphState`` plus an evaluation suite. Formalizing that bundle as a single object is what makes
environments *reproducible by construction*: every environment that fills this contract inherits
the same compiled execution, ``vmap``-over-seeds batching, trajectory tracing, and metric wiring
for free, so adding a new one is "implement the five fields, register a name" rather than "wire a
bespoke run loop".

    EnvSpec ::= {
      config   : frozen params (closed over by the transforms; a dial meant to be swept
                 is seeded into global_attrs as a 0-d array instead, see run_grid)
      init_fn  : key -> GraphState                  (per-seed initial state; vmap-friendly)
      round_fn : (state, t, key) -> state           (one tick; == core.scan.RoundFn)
      trace_fn : state -> dict                       (per-step readout for trajectories)
      metrics  : { name : (trace -> scalar) }        (offline evaluation, e.g. the GovSim suite)
    }

``run`` / ``run_batch`` are thin wrappers over ``core.scan`` — the *only* place execution lives —
so the substrate code never touches the scan plumbing. All four ``run*`` methods compile the scan
once per spec and call signature, so repeat calls skip the compile, and every seed's trajectory
stays bit-identical to the un-jitted path (``EnvSpec._compiled``, ``tests/test_spec_jit.py``).
Governance and measurement compose on top by wrapping ``round_fn`` and extending ``trace_fn`` /
``metrics``; the contract does not change.
"""
from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax

from cilib.core.graph import GraphState
from cilib.core.reduce import (Reducer, run_scan_reduce, run_scan_reduce_batch,
                               run_scan_reduce_over)
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
        single ``run`` and one seed of a ``run_batch`` start from the same state and draw the same
        randomness, but they are not always the same trajectory. ``vmap`` scans a seed-constant
        field (a fixed network) unbatched, XLA sums it in a different order, and the float32
        difference can cross a discrete rule and grow: in 6 of 11 default environments the two
        differ, by up to 0.03 in ``ledger_society`` at T=50 (measured 2026-09-30). This predates
        compilation; ``run_batch`` keeps the old batch numbers rather than the per-seed ones.
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
        traced for ``0.0``; Python scalars there must not vary per call. Array
        entries of ``global_attrs`` are runtime inputs like any other leaf, which
        is what lets ``run_grid`` vary them without a new program.
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
        n_steps = operator.index(n_steps)
        items = tuple(sorted(reducers.items()))
        k_init, k_run = jr.split(key)
        state = self.init_fn(k_init)
        accs = tuple(r.init(state, n_steps) for _, r in items)
        if _traced((state, accs, k_run)):
            return run_scan_reduce(self.round_fn, state, n_steps, k_run, reducers,
                                   trace_fn=trace_fn)

        def scan(state, accs, k_run, ts):
            return run_scan_reduce_over(self.round_fn, state, accs, ts, k_run, items,
                                        n_steps, trace_fn)
        final_state, final_accs, trace = self._compiled(
            "reduce", scan, (state, accs, k_run, jnp.arange(n_steps)), trace_fn, static=items)
        return final_state, trace, _finalize(items, final_accs, n_steps)

    def run_reduced_batch(self, key: Any, n_seeds: int, n_steps: int,
                          reducers: Dict[str, Reducer],
                          trace_fn: Optional[TraceFn] = None
                          ) -> Tuple[GraphState, Any, Dict[str, Any]]:
        """``vmap``ped sweep with in-scan reduction. ``reduced[name]`` is shaped
        ``(n_seeds,)`` for a scalar metric — already the per-seed vector the
        benchmark harness bootstraps over, with no trajectory kept."""
        n_steps = operator.index(n_steps)
        items = tuple(sorted(reducers.items()))
        keys = jr.split(key, n_seeds)

        def build(k):
            state = self.init_fn(k)
            return state, tuple(r.init(state, n_steps) for _, r in items)
        batch = _init_batch(build, keys) if n_seeds and not _traced(keys) else None
        if batch is None:
            return run_scan_reduce_batch(self.round_fn, self.init_fn, n_steps, keys,
                                         reducers, trace_fn=trace_fn)
        leaves, treedef, axes, k_run = batch

        def scan(leaves, k_run, ts):
            def one(ls, k):
                state, accs = jax.tree_util.tree_unflatten(treedef, ls)
                return run_scan_reduce_over(self.round_fn, state, accs, ts, k, items,
                                            n_steps, trace_fn)
            return jax.vmap(one, in_axes=(list(axes), 0))(leaves, k_run)
        final_states, final_accs, traces = self._compiled(
            "reduce_batch", scan, (leaves, k_run, jnp.arange(n_steps)), trace_fn,
            static=(items, treedef, axes))
        return final_states, traces, jax.vmap(lambda a: _finalize(items, a, n_steps))(final_accs)

    def run_grid(self, key: Any, n_seeds: int, n_steps: int, grid: Dict[str, Any],
                 reducers: Dict[str, Reducer], trace_fn: Optional[TraceFn] = None,
                 batch: int = 1) -> Tuple[GraphState, Any, Dict[str, Any]]:
        """``run_reduced_batch`` over a grid of dial values, compiled once per grid shape.

        ``grid`` maps ``global_attrs`` keys to arrays with a leading cell axis ``G``:
        cell ``g`` runs every seed with ``global_attrs[k] = grid[k][g]``. The dial
        values reach the program as runtime arguments, so a new set of values of the
        same shape reuses the compiled program; looping ``make_env`` over a grid pays
        one compile per cell. Every cell uses the same ``key``, so seed ``s`` starts
        from the same state and draws the same randomness in every cell (common random
        numbers): compare cells as paired differences over seeds.

        A grid key must already be a ``jax.Array`` in the initial ``global_attrs`` (a
        dial lifted from closed-over config, the ``*_now`` pattern in
        ``ledger_society/state.py``); grid values are cast to that entry's dtype, and
        ``step`` and ``rng_key`` are refused. What is overridden is the entry's value
        in the initial state, nothing more. It acts as a dial only if ``round_fn``
        reads it and nothing writes it: a global a lever rewrites each tick keeps the
        grid value for one round, and ``init_fn`` ran with the config value, so a
        dial that also shapes the initial state cannot be swept this way.

        Cells run inside one program as a ``lax.map``, ``batch`` cells at a time.
        With ``batch=1`` each cell is the ``run_reduced_batch`` program for its value:
        the trajectory and every reducer value the scan writes are bit-identical to
        that call on a spec whose config holds the value. One exception: reducer
        accumulators are built inside the program, from the state after the
        override, and an ``init`` value the scan never overwrites (``at_step`` at
        onset 0 or past the run, ``last`` at T=0) can differ by one float32 ulp,
        because XLA folds constants the eager reference computes op by op.
        ``batch > 1`` vectorizes that many cells, which batches whatever the dial
        touches; XLA may then sum in a different order and the numbers move at
        float32 rounding, enough to cross a threshold in some models
        (docs/run-grid-design.md has the measured gap and the speed trade).

        The program is cached on the reducer and ``trace_fn`` objects themselves:
        build them once and reuse them across calls, or every call recompiles.

        Args:
            grid: ``{global_attrs key: array (G, *entry_shape)}``, ``G >= 1``.
            batch: cells vectorized together, ``>= 1``. Part of the program, so a
                new ``batch`` is a new compile.

        Returns:
            ``(final_states, traces, reduced)``, each with leading axes ``(G, n_seeds)``.
        """
        n_steps = operator.index(n_steps)
        batch = operator.index(batch)
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        items = tuple(sorted(reducers.items()))
        names = tuple(sorted(grid))
        keys = jr.split(key, n_seeds)
        cells = _grid_cells(jax.eval_shape(self.init_fn, jr.split(key)[0]), grid, names)
        init = (_init_batch(self.init_fn, keys)
                if n_seeds and not _traced((keys, cells)) else None)
        if init is None:
            def cell(g):
                def one(k):
                    k_init, k_run = jr.split(k)
                    state = with_globals(self.init_fn(k_init), g)
                    return run_scan_reduce(self.round_fn, state, n_steps, k_run, reducers,
                                           trace_fn=trace_fn)
                return jax.vmap(one)(keys)
            return _map_cells(cell, cells, batch)
        leaves, treedef, axes, k_run = init

        def scan(leaves, k_run, cells, ts):
            def cell(g):
                def one(ls, k):
                    state = with_globals(jax.tree_util.tree_unflatten(treedef, ls), g)
                    accs = tuple(r.init(state, n_steps) for _, r in items)
                    return run_scan_reduce_over(self.round_fn, state, accs, ts, k, items,
                                                n_steps, trace_fn)
                return jax.vmap(one, in_axes=(list(axes), 0))(leaves, k_run)
            return _map_cells(cell, cells, batch)
        final_states, final_accs, traces = self._compiled(
            "grid", scan, (leaves, k_run, cells, jnp.arange(n_steps)), trace_fn,
            static=(items, treedef, axes, names, batch))
        finalize = jax.vmap(jax.vmap(lambda a: _finalize(items, a, n_steps)))
        return final_states, traces, finalize(final_accs)

    def evaluate(self, trace: Any) -> Dict[str, Any]:
        """Score a trajectory (or batch of trajectories) with every metric in the suite."""
        return {name: fn(trace) for name, fn in self.metrics.items()}


def _finalize(items, accs, n_steps):
    return {name: r.finalize(a, n_steps) for a, (name, r) in zip(accs, items)}


def _map_cells(f, cells, batch):
    """``f`` over the leading cell axis, ``batch`` cells vectorized at a time."""
    return lax.map(f, cells, batch_size=None if batch == 1 else batch)


def with_globals(state: GraphState, overrides: Dict[str, Any]) -> GraphState:
    """``state`` with existing array entries of ``global_attrs`` replaced. Pure.

    Only an entry that is already an array can be replaced: a Python scalar there
    is static pytree aux, and swapping it would change the program, not its inputs.
    """
    for k in overrides:
        if not isinstance(state.global_attrs.get(k), jax.Array):
            raise KeyError(f"global_attrs[{k!r}] is not an array entry of this state")
    return state.replace(global_attrs={**state.global_attrs, **overrides})


def _grid_cells(state_shape: GraphState, grid: Dict[str, Any], names) -> Dict[str, Any]:
    """Check a ``run_grid`` grid against the initial state and cast it to its dtypes.

    ``state_shape`` is ``jax.eval_shape(init_fn, key)``: array entries of
    ``global_attrs`` come back as ``ShapeDtypeStruct``, static ones as themselves.
    """
    if not grid:
        raise ValueError("run_grid needs at least one grid key")
    clock = sorted(set(grid) & {"step", "rng_key"})
    if clock:
        raise ValueError(f"{clock} cannot be grid keys: the round rewrites rng_key and "
                         "counts with step, so a grid there is ignored or shifts the clock")
    cells, sizes = {}, set()
    for k in names:
        entry = state_shape.global_attrs.get(k)
        if not isinstance(entry, jax.ShapeDtypeStruct):
            raise ValueError(
                f"grid key {k!r} is not a jax.Array in the initial global_attrs, so its "
                "value is closed over (or static) and cannot vary at runtime. Lift it: "
                "seed global_attrs[k] = jnp.array(cfg.x, dtype) in init_fn and read it "
                "there in round_fn (the *_now pattern, ledger_society/state.py).")
        if entry.weak_type:
            raise ValueError(
                f"global_attrs[{k!r}] is weakly typed; seed it with an explicit dtype "
                "so a grid value cannot change type promotion")
        v = jnp.asarray(grid[k], dtype=entry.dtype)
        if v.ndim < 1 or v.shape[1:] != entry.shape:
            raise ValueError(f"grid[{k!r}] has shape {v.shape}; expected "
                             f"(G, *{entry.shape}) for the cell axis G")
        raw = grid[k]
        if (jnp.issubdtype(entry.dtype, jnp.integer) and not _traced(raw)
                and not np.array_equal(np.asarray(raw, dtype=float), np.asarray(v, dtype=float))):
            raise ValueError(f"grid[{k!r}] has values that are not {entry.dtype} "
                             "integers; casting would change them")
        cells[k] = v
        sizes.add(v.shape[0])
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError(f"grid keys need one common, non-zero cell count; got {sorted(sizes)}")
    return cells


def _traced(tree) -> bool:
    """True inside a caller's ``jit`` / ``vmap`` / ``grad``: their trace compiles the whole
    call, so the ``run*`` methods take the old inline path and add no program of their own."""
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(tree))


def _init_batch(init_fn, keys):
    """Per-seed initial states (or any per-seed pytree ``init_fn`` builds, such as a state
    with its reducer accumulators), built eagerly, in the layout ``run_scan_batch`` scans them.

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
