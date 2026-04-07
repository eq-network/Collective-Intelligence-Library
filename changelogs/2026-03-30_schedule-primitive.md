# Schedule Primitive: Cadence as Experimental Variable

**Date:** 2026-03-30
**File:** `core/schedule.py` (new)
**Task:** Plan 1 Phase B1

## What Changed

New module implementing the whitepaper's Schedule primitive (§3.3). Each transform gets a cadence and phase offset. Transform fires at tick t iff `(t - phase_offset) % cadence == 0`.

## Why

From the whitepaper: "The schedule that orchestrates these mechanisms is itself a first-class experimental variable: the same mechanism composition under different temporal configurations can produce qualitatively different collective outcomes."

The key research question: how does democratic voting cadence (k=1 vs k=5 vs k=10) affect outcomes under adversarial pressure?

## Usage

```python
entries = [
    ScheduleEntry(market, cadence=1),        # every tick
    ScheduleEntry(network, cadence=1),        # every tick
    ScheduleEntry(democracy, cadence=k),      # every k ticks — experimental variable
    ScheduleEntry(harvest, cadence=1),
]
step_fn = make_scheduled_step(entries)
```

## JIT Compatibility

The naive `if should_fire(entry, tick):` breaks under JIT because `tick` is a traced value. The solution:

```python
def _maybe_apply(state, entry, tick):
    fire = (tick - entry.phase_offset) % entry.cadence == 0
    return jax.lax.cond(fire, entry.transform, lambda s: s, state)
```

`lax.cond` traces BOTH branches at compile time but executes only one at runtime. The `cadence` and `phase_offset` are Python ints (static), `tick` is a JAX value (dynamic). This works because JAX traces arithmetic on mixed static/dynamic operands.

## Schedule Diagram

```
tick:       0  1  2  3  4  5  6  7  8  9
market:     x  x  x  x  x  x  x  x  x  x
network:    x  x  x  x  x  x  x  x  x  x
democracy:  x        x        x        x
harvest:    x  x  x  x  x  x  x  x  x  x
```

## Design Decisions

- `make_scheduled_step` reads `state.global_attrs["step"]` and increments it after all entries fire
- The Python for-loop over entries is unrolled at JIT trace time (static number of entries)
- Compatible with `lax.scan`: the step function has fixed structure regardless of tick value
