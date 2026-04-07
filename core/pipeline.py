"""
Pipeline compiler — derive execution order from transform read/write sets.

Given transforms decorated with @transform(reads=..., writes=...),
build a dependency DAG, topologically sort into parallel batches,
and return a single composed Transform.

This is the typed DAG architecture from the whitepaper §3.2:
transforms declare which fields they read/write, and the pipeline
order is *derived*, not manually specified.
"""
from typing import List, Dict, Set

from .graph import GraphState
from .category import Transform, sequential, parallel


def _build_dependency_graph(
    transforms: List[Transform],
) -> Dict[int, Set[int]]:
    """
    Build a dependency DAG from read/write set intersections.

    Transform i depends on transform j if i.reads intersects j.writes
    (i needs something j produces).
    """
    n = len(transforms)
    deps: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i in range(n):
        reads_i = getattr(transforms[i], 'reads', frozenset())
        for j in range(n):
            if i == j:
                continue
            writes_j = getattr(transforms[j], 'writes', frozenset())
            if reads_i & writes_j:
                deps[i].add(j)

    return deps


def _detect_write_conflicts(transforms: List[Transform]) -> List[str]:
    """Check for two transforms that write the same field."""
    errors = []
    n = len(transforms)
    for i in range(n):
        for j in range(i + 1, n):
            wi = getattr(transforms[i], 'writes', frozenset())
            wj = getattr(transforms[j], 'writes', frozenset())
            overlap = wi & wj
            if overlap:
                ni = getattr(transforms[i], 'name', f'transform_{i}')
                nj = getattr(transforms[j], 'name', f'transform_{j}')
                errors.append(
                    f"Write conflict: {ni} and {nj} both write {overlap}"
                )
    return errors


def _topological_batches(
    transforms: List[Transform],
    deps: Dict[int, Set[int]],
) -> List[List[int]]:
    """
    Kahn's algorithm with parallel batch grouping.

    Returns a list of batches, where each batch contains indices of
    transforms that can execute in parallel (all dependencies satisfied
    by prior batches).
    """
    n = len(transforms)
    in_degree = {i: len(deps[i]) for i in range(n)}

    # Reverse map: who depends on me?
    dependents: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i, dep_set in deps.items():
        for j in dep_set:
            dependents[j].add(i)

    ready = [i for i in range(n) if in_degree[i] == 0]
    batches = []
    processed = 0

    while ready:
        batch = sorted(ready)
        batches.append(batch)
        processed += len(batch)

        next_ready = []
        for i in batch:
            for dependent in dependents[i]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_ready.append(dependent)
        ready = next_ready

    if processed < n:
        in_cycle = [
            getattr(transforms[i], 'name', f'transform_{i}')
            for i in range(n) if in_degree[i] > 0
        ]
        raise ValueError(f"Cycle detected involving: {in_cycle}")

    return batches


def validate_pipeline(transforms: List[Transform]) -> List[str]:
    """
    Validate a set of transforms for pipeline compilation.
    Returns list of warnings/errors without raising.
    """
    issues = []

    for i, t in enumerate(transforms):
        name = getattr(t, 'name', f'transform_{i}')
        if not hasattr(t, 'reads') or not hasattr(t, 'writes'):
            issues.append(f"{name}: missing @transform decorator (no reads/writes)")

    issues.extend(_detect_write_conflicts(transforms))

    deps = _build_dependency_graph(transforms)
    try:
        _topological_batches(transforms, deps)
    except ValueError as e:
        issues.append(str(e))

    return issues


def get_execution_order(transforms: List[Transform]) -> List[List[Transform]]:
    """
    Return the execution plan as a list of parallel batches.

    Each batch is a list of transforms that can run simultaneously
    (all dependencies satisfied by prior batches).
    """
    deps = _build_dependency_graph(transforms)
    index_batches = _topological_batches(transforms, deps)
    return [[transforms[i] for i in batch] for batch in index_batches]


def compile_pipeline(transforms: List[Transform]) -> Transform:
    """
    Compile transforms into a single Transform with derived execution order.

    1. Check for write conflicts
    2. Build dependency DAG from read/write sets
    3. Topological sort into parallel batches
    4. Compose into a single Transform

    Returns:
        A single Transform (GraphState -> GraphState)

    Raises:
        ValueError: if write conflicts or cycles detected
    """
    conflicts = _detect_write_conflicts(transforms)
    if conflicts:
        raise ValueError("Pipeline has write conflicts:\n" + "\n".join(conflicts))

    batches = get_execution_order(transforms)

    # Each batch: single transform or parallel group (disjoint writes guaranteed)
    # Batches execute sequentially (inter-batch dependencies)
    batch_transforms = []
    for batch in batches:
        if len(batch) == 1:
            batch_transforms.append(batch[0])
        else:
            batch_transforms.append(parallel(*batch))

    compiled = sequential(*batch_transforms)

    # Attach aggregate metadata
    all_reads = frozenset().union(*(getattr(t, 'reads', frozenset()) for t in transforms))
    all_writes = frozenset().union(*(getattr(t, 'writes', frozenset()) for t in transforms))
    compiled.reads = all_reads
    compiled.writes = all_writes
    compiled.name = "compiled_pipeline"
    compiled._batches = batches  # For inspection

    return compiled
