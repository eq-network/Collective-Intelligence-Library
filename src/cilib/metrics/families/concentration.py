"""
Concentration metrics: inequality and market-power readouts.

Generic building blocks for the disempowerment scenarios (alpha plan: ``gini``, ``hhi``
as reusable in-loop readouts). ``gini_of`` / ``hhi_of`` are bare-array helpers usable on
any non-negative 1-D distribution; the GraphState wrappers below bind them to
``node_attrs["last_harvest"]`` for the commons environments.

All functions are pure and stay in jnp (no ``float()``) for JIT/vmap compatibility.
"""
import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.metrics.identity import kind


def gini_of(values):
    """Gini coefficient of a non-negative 1-D array. 0 = perfect equality, ->1 = maximal.

    Sorted-order formulation: exact and JIT-safe, no O(N^2) pairwise matrix.
    """
    v = jnp.sort(values)
    n = v.shape[0]
    index = jnp.arange(1, n + 1)
    total = jnp.sum(v)
    numerator = 2.0 * jnp.sum(index * v) - (n + 1) * total
    denom = n * total
    return jnp.where(denom > 0, numerator / denom, 0.0)


def hhi_of(values):
    """Herfindahl-Hirschman index of a non-negative 1-D array.

    Sum of squared shares: 1/N for an equal split, 1.0 for a monopoly, 0.0 for an
    all-zero distribution (no activity to concentrate).
    """
    total = jnp.sum(values)
    shares = values / (total + 1e-12)
    return jnp.sum(shares ** 2)


def centralization_of(values):
    """Freeman-style centralization of a non-negative 1-D distribution:
    ``(max_share - 1/N) / (1 - 1/N)``. 0 = perfectly dispersed, 1 = one node
    holds everything. The A4 mainline indicator (alpha plan: "centralization"),
    reading how far the top actor is above an equal split."""
    n = values.shape[0]
    total = jnp.sum(values)
    max_share = jnp.max(values) / (total + 1e-12)
    return jnp.where(total > 0, (max_share - 1.0 / n) / (1.0 - 1.0 / n), 0.0)


@kind("concentration")
def harvest_gini(state: GraphState):
    """Gini of per-agent last harvest. 0 = equal extraction."""
    return gini_of(state.node_attrs["last_harvest"])


@kind("concentration")
def harvest_hhi(state: GraphState):
    """HHI of per-agent last harvest. 1/N = equal extraction, 1 = one agent takes all."""
    return hhi_of(state.node_attrs["last_harvest"])


CONCENTRATION_METRICS = {
    "harvest_gini": harvest_gini,
    "harvest_hhi": harvest_hhi,
}
