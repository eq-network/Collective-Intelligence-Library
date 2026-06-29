"""
Pipeline compiler — derive execution structure from transform read/write sets.

Given transforms decorated with ``@transform(reads=..., writes=...)``, build a
dependency DAG from data hazards, topologically sort into parallel batches, and
return a single composed Transform.

**Hazards are oriented by program order.** For two transforms i < j (i precedes j in
the list), j depends on i if they share a field with at least one write between them:

    RAW  (read-after-write):   writes_i ∩ reads_j   — j consumes what i produced
    WAR  (write-after-read):   reads_i  ∩ writes_j   — j must not clobber before i reads
    WAW  (write-after-write):  writes_i ∩ writes_j   — the later write must win

Orienting every hazard *forward* in program order makes the graph acyclic by
construction and guarantees the derived schedule is **behaviour-identical** to
running the list in order: any two transforms with no hazard between them touch
disjoint state (or share only reads), so they commute and may run in parallel. This
is why a stateful round — sequential read-modify-write of a shared stock, or an agent
that reads an old policy and writes a new one — compiles correctly: those become
plain forward edges, not conflicts or cycles. The compiler's job is to *recover the
parallelism*, not to reorder across real dependencies.
"""
from typing import List, Dict, Set

from .graph import GraphState
from .category import Transform, sequential, parallel


def _build_dependency_graph(transforms: List[Transform]) -> Dict[int, Set[int]]:
    """Edges from RAW / WAR / WAW hazards, oriented by program order (j after i)."""
    n = len(transforms)
    deps: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i in range(n):
        wi = getattr(transforms[i], 'writes', frozenset())
        ri = getattr(transforms[i], 'reads', frozenset())
        for j in range(i + 1, n):
            wj = getattr(transforms[j], 'writes', frozenset())
            rj = getattr(transforms[j], 'reads', frozenset())
            raw = wi & rj
            war = ri & wj
            waw = wi & wj
            if raw or war or waw:
                deps[j].add(i)          # j runs after i

    return deps


def _topological_batches(
    transforms: List[Transform],
    deps: Dict[int, Set[int]],
) -> List[List[int]]:
    """
    Kahn's algorithm with parallel batch grouping.

    Each batch contains transform indices whose dependencies are all satisfied by
    prior batches; because every hazard is an edge, two transforms in the same batch
    have no hazard between them (disjoint writes, no cross read) and so commute.
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
    """Return a list of issues (missing decorators, cycles) without raising."""
    issues = []

    for i, t in enumerate(transforms):
        name = getattr(t, 'name', f'transform_{i}')
        if not hasattr(t, 'reads') or not hasattr(t, 'writes'):
            issues.append(f"{name}: missing @transform decorator (no reads/writes)")

    deps = _build_dependency_graph(transforms)
    try:
        _topological_batches(transforms, deps)
    except ValueError as e:
        issues.append(str(e))

    return issues


def get_execution_order(transforms: List[Transform]) -> List[List[Transform]]:
    """Return the execution plan as a list of parallel batches."""
    deps = _build_dependency_graph(transforms)
    index_batches = _topological_batches(transforms, deps)
    return [[transforms[i] for i in batch] for batch in index_batches]


def compile_pipeline(transforms: List[Transform]) -> Transform:
    """
    Compile transforms into a single Transform with derived execution structure.

    1. Build a hazard DAG from read/write sets (oriented by program order)
    2. Topologically sort into parallel batches
    3. Compose: each batch is a single transform or a ``parallel`` group, batches
       run in sequence

    The result is behaviour-identical to ``sequential(*transforms)`` but executes
    independent steps as parallel groups.

    Raises:
        ValueError: if a cycle is somehow present (not possible for decorated,
        program-ordered transforms — only if dependencies are injected by hand).
    """
    batches = get_execution_order(transforms)

    # Each batch: a single transform, or a parallel group (no intra-batch hazard,
    # so writes are disjoint and execution order within the batch is irrelevant).
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
