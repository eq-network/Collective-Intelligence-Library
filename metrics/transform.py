"""
Auto-generate a metrics Transform from a dict of metric functions.

Each metric function is GraphState -> JAX scalar. The generated transform
writes values into pre-allocated arrays in global_attrs via .at[step].set(),
which is fully JIT and lax.scan compatible.

Usage:
    metrics = {**ECONOMIC_METRICS, **GOVERNANCE_METRICS}
    # Pre-allocate in state factory:
    for name in metrics: global_attrs[f"metric_{name}"] = jnp.zeros(T)
    # Append to pipeline:
    pipeline = sequential(..., make_metrics_transform(metrics))
"""
from typing import Callable, Dict

from core.graph import GraphState
from core.category import Transform


def make_metrics_transform(metrics: Dict[str, Callable[[GraphState], float]]) -> Transform:
    """Create a transform that fills pre-allocated metric arrays in global_attrs.

    The for-loop over metrics unrolls at JAX trace time (static dict iteration),
    so each metric becomes a traced operation in the computation graph.

    Requires:
        global_attrs must contain 'metric_{name}': jnp.zeros(T) for each metric.

    Args:
        metrics: dict mapping metric name to a pure function GraphState -> scalar.

    Returns:
        Transform that writes all metric values at the current step index.
    """
    # Freeze the dict into a tuple for deterministic ordering
    metric_items = tuple(sorted(metrics.items()))

    def transform(state: GraphState) -> GraphState:
        step = state.global_attrs["step"]
        for name, fn in metric_items:
            value = fn(state)
            key = f"metric_{name}"
            arr = state.global_attrs[key]
            state = state.update_global_attr(key, arr.at[step].set(value))
        return state

    transform.name = "metrics"
    transform.reads = frozenset()  # reads everything (conservative)
    transform.writes = frozenset(f"metric_{name}" for name, _ in metric_items)
    return transform
