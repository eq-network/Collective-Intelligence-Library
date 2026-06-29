"""Collective Intelligence Library (CI Lib).

Composable mechanism simulation via functional graph transforms. The library is a
set of typed *catalogs* of swappable building blocks — agents, transformations,
mechanisms, environments — composed into pipelines and run on a JAX scan tier.

See ARCHITECTURE.md for the pattern map and EXTENDING.md for how to add a block.
"""

__version__ = "0.1.0"

from cilib.core import (
    GraphState,
    Transform,
    transform,
    sequential,
    compile_pipeline,
    run_scan,
    run_scan_batch,
)

__all__ = [
    "__version__",
    "GraphState",
    "Transform",
    "transform",
    "sequential",
    "compile_pipeline",
    "run_scan",
    "run_scan_batch",
]
