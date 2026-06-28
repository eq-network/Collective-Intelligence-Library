"""
Pure compiled simulation loop — the "scan tier".

This is the FAST path. When every agent (and the world) is PURE — no genuine
runtime side effects — a single round of the simulation is a pure
``GraphState -> GraphState`` function, so running it for ``T`` steps compiles to
one ``jax.lax.scan`` and ``vmap``s over seeds / swept dials as a single program.
That is what makes "thousands of repeated simulations" cheap.

Contrast ``core/time.py`` (``World`` / ``tick_world`` / ``run_n_ticks``), the
EAGER tier, which is the right home for genuinely effectful agents (the
``LLMAgent`` HTTP port), event logs with duration, and host-side prediction
resolution. Those cannot live inside a compiled scan and should not try to.

Conventions for the scan tier (violating these breaks compilation):

- **Time is the scan index**, never stored in ``global_attrs``. ``global_attrs``
  is *static* pytree aux (see ``GraphState.tree_flatten``): anything placed there
  is baked in at trace time and cannot change across steps. Per-step-changing
  data (the tick, swept dials) must be a traced array or threaded through the
  carry — never a global attr.
- **Randomness is threaded**: each step splits a fresh key from the carry key
  (the NumPyro pattern). No global RNG.
- **Fixed structure & shapes**: ``round_fn`` must return a ``GraphState`` with the
  same pytree structure and the same array shapes/dtypes it received (``scan``
  requires the carry to be shape-stable). Use fixed-capacity state
  (``core.graph.create_padded_state``) + a boolean active mask
  (``GraphState.get_active_mask``); never call ``num_nodes`` /
  ``get_active_indices`` inside the scanned body (they host-sync / return
  dynamically shaped arrays and are illegal under jit/scan).
"""
from typing import Callable, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jax import lax, random

from .graph import GraphState

# A pure round of the simulation: (state, t, key_t) -> state.
# `t` is the (traced) scan index; `key_t` is a fresh per-step PRNG key.
RoundFn = Callable[[GraphState, jnp.ndarray, Any], GraphState]

# Optional per-step readout recorded into the trace: state -> pytree.
TraceFn = Callable[[GraphState], Any]


def run_scan(
    round_fn: RoundFn,
    init_state: GraphState,
    n_steps: int,
    key: Any,
    trace_fn: Optional[TraceFn] = None,
) -> Tuple[GraphState, Any]:
    """Run ``round_fn`` for ``n_steps`` as a single compiled ``lax.scan``.

    Args:
        round_fn: Pure ``(state, t, key_t) -> state``. Must preserve the pytree
            structure and all array shapes/dtypes of ``state``.
        init_state: Initial ``GraphState``. Only its array children change across
            steps; ``global_attrs`` stays fixed (it is static aux).
        n_steps: Number of steps (a static Python int — it is the scan length).
        key: A PRNG key.
        trace_fn: Optional ``state -> pytree`` recorded after every step and
            stacked along a leading time axis of length ``n_steps`` in the
            returned trace. If ``None``, the returned trace is ``None``.

    Returns:
        ``(final_state, trace)``. ``trace`` is the stacked per-step ``trace_fn``
        output (leading axis = ``n_steps``), or ``None`` when ``trace_fn`` is
        ``None``.
    """
    def body(carry, t):
        state, k = carry
        k, step_key = random.split(k)
        new_state = round_fn(state, t, step_key)
        out = trace_fn(new_state) if trace_fn is not None else None
        return (new_state, k), out

    (final_state, _), trace = lax.scan(
        body, (init_state, key), jnp.arange(n_steps)
    )
    return final_state, trace


def run_scan_batch(
    round_fn: RoundFn,
    init_fn: Callable[[Any], GraphState],
    n_steps: int,
    keys: Any,
    trace_fn: Optional[TraceFn] = None,
) -> Tuple[GraphState, Any]:
    """``vmap`` ``run_scan`` over a batch of PRNG keys — the sweep fast path.

    Each seed gets its own initial state (built by ``init_fn``) and its own run,
    but the whole batch compiles and executes as one program. This is how a
    multi-seed / multi-replication sweep becomes a single ``vmap(scan(...))``
    instead of N separate runs.

    Args:
        round_fn: Pure round, as in :func:`run_scan`.
        init_fn: ``key -> GraphState`` building one seed's initial state. Must
            return the same pytree structure / shapes for every key.
        n_steps: Static number of steps.
        keys: Batch of PRNG keys, shape ``(B, ...)`` (e.g. from
            ``jax.random.split(key, B)``).
        trace_fn: Optional per-step readout (see :func:`run_scan`).

    Returns:
        ``(final_states, traces)`` batched along a leading axis of length ``B``.
        ``traces`` has shape ``(B, n_steps, ...)`` or ``None``.
    """
    def one(k):
        k_init, k_run = random.split(k)
        init_state = init_fn(k_init)
        return run_scan(round_fn, init_state, n_steps, k_run, trace_fn)

    return jax.vmap(one)(keys)
