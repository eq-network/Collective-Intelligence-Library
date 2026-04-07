# Pipeline Compiler: Derived Execution Order from Typed DAG

**Date:** 2026-03-30
**File:** `core/pipeline.py` (new)
**Task:** Plan 1 Phase A2

## What Changed

New module that takes `@transform`-decorated functions and derives execution order by building a dependency DAG from read/write sets.

## Why

From the whitepaper §3.2: "Each transform declares its read set and write set. These declarations enable the system to determine which transforms can safely execute in parallel and to verify pipeline well-formedness."

Manual ordering via `sequential(vote, harvest, reward)` is error-prone and doesn't scale. The compiler derives the ordering automatically.

## How It Works

1. **Build dependency DAG:** Transform A depends on B if `A.reads ∩ B.writes ≠ ∅`
2. **Detect write conflicts:** Two transforms writing the same field → error
3. **Topological sort with batching (Kahn's algorithm):** Collect all zero-in-degree nodes into one batch. Transforms within a batch have no dependency → run via `parallel()`. Batches execute via `sequential()`.
4. **Return composed Transform**

## Example

```python
pipeline = compile_pipeline([harvest, reward, network, vote])  # any order

# Derived execution plan:
#   Batch 0: [vote, network]   ← parallel (disjoint writes)
#   Batch 1: [harvest]         ← depends on vote (reads harvest_target)
#   Batch 2: [reward]          ← depends on harvest (reads last_harvest)
```

Nobody specified this ordering. It was derived from:
- `vote.writes = {harvest_target}`, `harvest.reads = {harvest_target, resource_level}` → harvest depends on vote
- `harvest.writes = {last_harvest}`, `reward.reads = {last_harvest}` → reward depends on harvest
- `network.reads = {signal}`, `network.writes = {received_signals}` → independent of vote/harvest/reward

## API

```python
compile_pipeline(transforms) -> Transform     # main entry point
get_execution_order(transforms) -> [[Transform]]  # inspect batch plan
validate_pipeline(transforms) -> [str]        # check without raising
```

## Design Decisions

- Multi-transform batches use `parallel()` (disjoint writes guaranteed by the DAG structure)
- Single-transform batches use the transform directly (no wrapping overhead)
- Batches are composed via `sequential()` (inter-batch dependencies require ordering)
- The compiled pipeline carries `.reads`, `.writes`, `.name` metadata
