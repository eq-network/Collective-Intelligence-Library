"""
Temporal foundation for Mycorrhiza — the EAGER tier.

Time is discrete: a monotonically increasing tick counter.
Events happen at specific ticks and have duration.
The World is (tick, GraphState, EventLog).

TWO TIERS OF TIME
-----------------
This module (``World`` / ``Event`` / ``EventLog`` / ``tick_world`` /
``run_n_ticks``) is the **eager tier**. It is the right home for things that
cannot live inside a compiled loop: genuinely effectful agents (the
``LLMAgent`` HTTP port), append-only event logs with duration, and host-side
prediction resolution (``core/predictions.py``, which works over Python lists
of events and string ids). The ``run_n_ticks`` loop is a plain Python fold and
``tick`` is carried in ``global_attrs``.

For PURE simulations — where every agent and the world are side-effect free —
use the **pure compiled tier** in ``core/scan.py`` (``run_scan`` /
``run_scan_batch``) instead. There the whole run is one ``jax.lax.scan`` that
``vmap``s over seeds/dials as a single program, which is what makes large
sweeps cheap.

Important: ``GraphState.global_attrs`` is *static* pytree aux (see
``GraphState.tree_flatten``). Storing ``tick`` there, as ``tick_world`` does, is
fine on this eager tier but must NOT be relied upon inside ``jit``/``scan`` —
on the scan tier, time is the scan index and per-step state lives in the array
children, never in ``global_attrs``.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from .graph import GraphState

# Type aliases
Tick = int
NodeID = int


@dataclass(frozen=True)
class Event:
    """
    An event is something that happens at a specific tick.

    Events have duration: they are created at tick T and complete at tick T+duration.
    """
    tick: Tick                      # When this event was created
    source: NodeID                  # Who initiated it
    target: NodeID                  # Who receives it (can be same as source)
    event_type: str                 # "message", "action", "prediction", "resolution"
    payload: Dict[str, Any]         # Event-specific data
    duration: Tick = 1              # How many ticks until it completes (default: 1)

    @property
    def completion_tick(self) -> Tick:
        """The tick at which this event completes."""
        return self.tick + self.duration

    def is_complete_at(self, current_tick: Tick) -> bool:
        """Check if this event has completed by the given tick."""
        return current_tick >= self.completion_tick


@dataclass(frozen=True)
class EventLog:
    """
    Append-only log of events.

    Events are either completed (visible in the system) or pending (still in progress).
    """
    completed: List[Event]
    pending: List[Event]

    def append(self, event: Event) -> 'EventLog':
        """
        Add a new event to the log.

        If duration is 0, the event is immediately completed.
        Otherwise, it starts as pending.
        """
        if event.duration == 0:
            return EventLog(
                completed=self.completed + [event],
                pending=self.pending
            )
        else:
            return EventLog(
                completed=self.completed,
                pending=self.pending + [event]
            )

    def complete_by_tick(self, current_tick: Tick) -> Tuple['EventLog', List[Event]]:
        """
        Move events from pending to completed if they finish at current_tick.

        Returns:
            (new_log, events_that_completed)
        """
        completing = [e for e in self.pending if e.completion_tick == current_tick]
        still_pending = [e for e in self.pending if e.completion_tick > current_tick]

        new_log = EventLog(
            completed=self.completed + completing,
            pending=still_pending
        )

        return new_log, completing

    def filter_completed(
        self,
        event_type: Optional[str] = None,
        source: Optional[NodeID] = None,
        target: Optional[NodeID] = None
    ) -> List[Event]:
        """Filter completed events by type, source, or target."""
        events = self.completed

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        if source is not None:
            events = [e for e in events if e.source == source]
        if target is not None:
            events = [e for e in events if e.target == target]

        return events


@dataclass(frozen=True)
class World:
    """
    The complete state of the system at a given tick.

    World = (tick, GraphState, EventLog)
    """
    tick: Tick
    state: GraphState
    log: EventLog


def tick_world(world: World) -> World:
    """
    Advance time by one tick.

    Process:
    1. Increment the tick counter
    2. Complete any events whose time has come
    3. Apply effects of completed events to state

    This is a pure function: World → World
    """
    new_tick = world.tick + 1

    # Complete pending events that finish at this tick
    new_log, completing_events = world.log.complete_by_tick(new_tick)

    # Update the tick in the state
    new_state = world.state.update_global_attr("tick", new_tick)

    # Apply effects of completed events
    # (For now, this is a stub - event semantics defined elsewhere)
    for event in completing_events:
        new_state = apply_event(new_state, event)

    return World(
        tick=new_tick,
        state=new_state,
        log=new_log
    )


def apply_event(state: GraphState, event: Event) -> GraphState:
    """
    Apply an event's effects to the GraphState.

    This is where event semantics are defined.
    Different event types have different effects.

    For now, this is a stub that can be extended.
    """
    # TODO: Implement event semantics
    # - "message": update edge attributes with message content
    # - "action": update node attributes based on action
    # - "prediction": record prediction in state
    # - "resolution": compute Brier score, update calibration

    return state


def run_n_ticks(world: World, n: int) -> World:
    """
    Run the world for N ticks.

    Pure functional loop - no side effects.
    """
    current = world
    for _ in range(n):
        current = tick_world(current)
    return current
