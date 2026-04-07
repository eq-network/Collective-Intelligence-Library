# Composition Operators: parallel() and conditional()

**Date:** 2026-03-30
**File:** `core/category.py`
**Task:** Plan 1 Phase C1, C2

## What Changed

Added two new composition operators alongside the existing `sequential()`:

1. **`parallel(*transforms)`** — execute on same input state, merge disjoint writes
2. **`conditional(predicate, transform, predicate_reads)`** — gate on a boolean condition

## Why

From the whitepaper §3.4: "Three operators combine transforms. Sequential executes in order. Parallel executes independently on the same input state and merges results. Conditional gates a transform on a predicate."

## parallel()

```python
both = parallel(market, network)
# market and network receive the SAME input state
# market writes: {allocations, budget, clearing_price}
# network writes: {received_signals}
# disjoint → safe merge
result = both(state)
```

How the merge works:
1. Apply each transform independently to the input state
2. Start from input state
3. For each transform's declared writes, take those fields from that transform's output
4. Combine into final output

Raises `ValueError` if write sets overlap (e.g., two transforms both write `budget`).

## conditional()

```python
cond_democracy = conditional(
    lambda s: s.global_attrs['resource_level'] > 1000.0,
    democracy,
    predicate_reads=['resource_level']
)
```

Uses `jax.lax.cond` for JIT compatibility. Both branches (transform and identity) are traced at compile time, only one executes at runtime.

**Constraint:** Both branches must return GraphStates with the same pytree structure (same set of dynamic global keys). This means conditional transforms should UPDATE existing fields, not ADD new ones.

The `predicate_reads` parameter tells the pipeline compiler what the predicate function reads, so it can correctly order this conditional relative to transforms that write those fields.

## Bug Fixes in This Session

| Bug | Fix |
|-----|-----|
| `sequential()` returned `identity` (factory) not `identity()` (transform) | Changed to `return identity()` |
| `identity().preserves = "ALL_PROPERTIES"` crashed `compose()` | Added sentinel check before `.intersection()` |
| `compose()` lost `.reads`/`.writes` metadata | Now propagates as union through chain |
| Pipeline compiler used `sequential()` for batches | Swapped to `parallel()` for multi-transform batches |
| `conditional()` omitted predicate reads from metadata | Added `predicate_reads` parameter |
