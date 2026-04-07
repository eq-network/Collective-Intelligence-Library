"""
Composable metrics system for Mycorrhiza simulations.

Metrics are pure functions (GraphState -> scalar) organized into families
by theoretical lens (economic, governance, graph theory, etc.).

A metrics Transform auto-generated from a dict of metric functions writes
scalars into pre-allocated arrays in global_attrs at each step, compatible
with JAX's lax.scan and vmap.
"""

from .transform import make_metrics_transform
from .export import write_trajectory_csv, write_summary_csv
from .families.economic import ECONOMIC_METRICS
from .families.governance import GOVERNANCE_METRICS
from .families.graph import GRAPH_METRICS

__all__ = [
    'make_metrics_transform',
    'write_trajectory_csv',
    'write_summary_csv',
    'ECONOMIC_METRICS',
    'GOVERNANCE_METRICS',
    'GRAPH_METRICS',
]
