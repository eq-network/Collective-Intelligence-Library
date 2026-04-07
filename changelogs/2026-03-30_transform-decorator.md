# @transform Decorator: Read/Write Set Metadata

**Date:** 2026-03-30
**File:** `core/category.py`
**Task:** Plan 1 Phase A1

## What Changed

Added `@transform(reads=[...], writes=[...])` decorator that attaches read/write metadata to pure functions.

## Why

The pipeline compiler needs to know which fields each transform reads and writes to derive execution order (topological sort of the dependency DAG). Without this, pipeline ordering must be specified manually via `sequential()`.

## Usage

```python
@transform(reads=["vote_weights"], writes=["harvest_target", "penalty_lambda"])
def direct_democracy(state: GraphState) -> GraphState:
    ...

# Metadata available:
direct_democracy.reads   # frozenset({'vote_weights'})
direct_democracy.writes  # frozenset({'harvest_target', 'penalty_lambda'})
direct_democracy.name    # 'direct_democracy'
```

## Design Decisions

- `.reads` and `.writes` are `frozenset[str]` — field names, not field paths
- The decorator wraps the function transparently — it's still `GraphState -> GraphState`
- Metadata is advisory, not enforced at runtime (enforcement is in the pipeline compiler)
- `compose()` propagates reads/writes through composition: `composed.reads = f.reads | g.reads`

## Also Fixed

- `compose()` now propagates `.reads`, `.writes`, `.name` through composition chains
- `identity().preserves = "ALL_PROPERTIES"` sentinel handled correctly in `compose()`
- `sequential()` empty case returns `identity()` (transform) not `identity` (factory)
