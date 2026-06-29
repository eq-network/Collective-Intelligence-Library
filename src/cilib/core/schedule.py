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
from typing import List
import jax
import jax.numpy as jnp

from .graph import GraphState
from .category import Transform


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
