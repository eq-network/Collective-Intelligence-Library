"""
Schedule primitive — map clock ticks to active transforms.

From the whitepaper §3.3: the Schedule is a global clock that ticks at a
base rate, with each transform assigned a cadence c_i and optional phase
offset φ_i. Transform τ_i fires at tick t if and only if (t - φ_i) mod c_i == 0.

This makes the timescale structure of a mechanism design explicit and
experimentally controllable. The schedule itself is a first-class
experimental variable.
"""
from dataclasses import dataclass
from typing import List, Optional, Union
import jax
import jax.numpy as jnp

from .graph import GraphState
from .category import Transform, conditional


@dataclass(frozen=True)
class ScheduleEntry:
    """A transform with its temporal scheduling parameters."""
    transform: Transform
    cadence: int = 1        # fire every c ticks
    phase_offset: int = 0   # fire when (t - φ) mod c == 0


def should_fire(entry: ScheduleEntry, tick: int) -> bool:
    """Check if a schedule entry should fire at this tick (eager, non-JIT)."""
    return (tick - entry.phase_offset) % entry.cadence == 0


def _maybe_apply(state: GraphState, entry: ScheduleEntry, tick) -> GraphState:
    """
    JIT-compatible conditional application of a scheduled transform.

    Uses jax.lax.cond: traces both branches but only executes one.
    The condition is computed from the traced tick value.
    """
    fire = (tick - entry.phase_offset) % entry.cadence == 0
    return jax.lax.cond(fire, entry.transform, lambda s: s, state)


def make_scheduled_step(entries: List[ScheduleEntry]) -> Transform:
    """
    Create a single Transform that conditionally executes scheduled entries.

    Reads state.global_attrs["step"] for the current tick.
    Applies each entry whose schedule says it fires at this tick.
    Increments the step counter after all entries are processed.

    Entries execute in the order given — for derived ordering, pass entries
    that have been sorted by compile_pipeline first.
    """
    def scheduled_step(state: GraphState) -> GraphState:
        tick = state.global_attrs["step"]

        for entry in entries:
            state = _maybe_apply(state, entry, tick)

        # Increment step counter
        new_step = tick + 1
        state = state.update_global_attr("step", new_step)
        return state

    # Aggregate metadata
    all_reads = frozenset(["step"]).union(
        *(getattr(e.transform, 'reads', frozenset()) for e in entries)
    )
    all_writes = frozenset(["step"]).union(
        *(getattr(e.transform, 'writes', frozenset()) for e in entries)
    )
    names = [getattr(e.transform, 'name', '?') for e in entries]
    scheduled_step.reads = all_reads
    scheduled_step.writes = all_writes
    scheduled_step.name = f"scheduled_step({', '.join(names)})"

    return scheduled_step


def scheduled(mechanism: Transform, cadence: int = 1, phase_offset: int = 0,
              onset: Union[int, str] = 0) -> Transform:
    """Attach a schedule to ONE mechanism: fires iff ``step >= onset`` AND
    ``(step - phase_offset) % cadence == 0``; identity otherwise (scan-safe lax.cond).

    This is the composition principle of the benchmark: mechanisms are pure rules,
    the schedule owns *when* they apply — "quota vote every 5 ticks", "taxation
    switching on at t=100" (``onset`` is the regime-shift dial for antifragility
    tests). Defaults are a transparent pass-through, so wrapping is opt-in.

    A string ``onset`` names a ``global_attrs`` entry read each tick instead of a
    fixed int, so the onset is runtime data and ``EnvSpec.run_grid`` can sweep it
    without recompiling. The environment must seed that entry (an int32 0-d array)
    in its initial state.

    Built on ``core.category.conditional``, which propagates ``.reads``/``.writes``
    (adding ``step`` to reads) so ``compile_pipeline`` still orders the wrapped
    mechanism correctly. (NOT built on ``gated``, which drops that metadata.)

    Complement of ``ScheduleEntry``/``make_scheduled_step`` (whole-round composition):
    ``scheduled`` wraps a single transform for use inside a ``compile_pipeline`` list.
    """
    if isinstance(onset, str):
        def fires(state: GraphState):
            step = state.global_attrs["step"]
            return ((step >= state.global_attrs[onset])
                    & (((step - phase_offset) % cadence) == 0))
        reads = ["step", onset]
    else:
        def fires(state: GraphState):
            step = state.global_attrs["step"]
            return (step >= onset) & (((step - phase_offset) % cadence) == 0)
        reads = ["step"]

    wrapped = conditional(fires, mechanism, predicate_reads=reads)
    wrapped.name = (f"scheduled({getattr(mechanism, 'name', '?')}, "
                    f"cadence={cadence}, phase={phase_offset}, onset={onset})")
    return wrapped


@dataclass(frozen=True)
class ScheduleSpec:
    """Serializable schedule dial for a benchmark condition entry. ``onset`` is a step
    or the name of a ``global_attrs`` entry holding one (see ``scheduled``)."""
    cadence: int = 1
    phase_offset: int = 0
    onset: Union[int, str] = 0


def apply_schedule(transform: Transform, spec: Optional[ScheduleSpec]) -> Transform:
    """Wrap ``transform`` per ``spec``; ``None`` means unscheduled (fires every tick)."""
    if spec is None:
        return transform
    return scheduled(transform, spec.cadence, spec.phase_offset, spec.onset)


def describe_schedule(entries: List[ScheduleEntry], num_ticks: int = 20) -> str:
    """
    Return a human-readable schedule diagram (whitepaper Figure 3 style).

    Example output:
        tick:       0  1  2  3  4  5  6  7  8  9
        market:     x  x  x  x  x  x  x  x  x  x
        network:    x  x  x  x  x  x  x  x  x  x
        democracy:  x        x        x        x
    """
    lines = []
    # Header
    tick_labels = [f"{t:>2}" for t in range(num_ticks)]
    name_width = max(len(getattr(e.transform, 'name', '?')) for e in entries)
    lines.append(f"{'tick':<{name_width}}  " + " ".join(tick_labels))

    for entry in entries:
        name = getattr(entry.transform, 'name', '?')
        marks = []
        for t in range(num_ticks):
            if (t - entry.phase_offset) % entry.cadence == 0:
                marks.append(" x")
            else:
                marks.append("  ")
        lines.append(f"{name:<{name_width}}  " + " ".join(marks))

    return "\n".join(lines)
