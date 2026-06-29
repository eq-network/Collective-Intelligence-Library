"""
Metric families organized by theoretical lens.

Each family is a dict of {name: GraphState -> JAX scalar}.
Compose by merging dicts: {**ECONOMIC_METRICS, **GOVERNANCE_METRICS}
"""
from .economic import ECONOMIC_METRICS
from .governance import GOVERNANCE_METRICS
from .graph import GRAPH_METRICS

__all__ = ['ECONOMIC_METRICS', 'GOVERNANCE_METRICS', 'GRAPH_METRICS']
