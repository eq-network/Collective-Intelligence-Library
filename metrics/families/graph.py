"""
Graph-theoretic metrics: spectral properties, connectivity, centrality.

Stubs for future expansion. These become valuable when analyzing
network mechanisms and delegation graph structure.

All functions are pure: GraphState -> JAX scalar.
"""
import jax.numpy as jnp

from core.graph import GraphState


def spectral_gap(state: GraphState):
    """Algebraic connectivity: 2nd-smallest eigenvalue of the graph Laplacian.

    Uses the trust_scores adjacency matrix. Higher = more connected.
    Computed via eigendecomposition — O(N^3), use at lower cadence for large N.
    """
    trust = state.node_attrs["trust_scores"]
    degree = jnp.sum(trust, axis=1)
    laplacian = jnp.diag(degree) - trust
    eigenvalues = jnp.linalg.eigvalsh(laplacian)
    return eigenvalues[1]  # 2nd smallest (Fiedler value)


def mean_degree(state: GraphState):
    """Mean weighted degree from the interaction adjacency matrix."""
    adj = state.adj_matrices["interaction"]
    degrees = jnp.sum(adj, axis=1)
    return jnp.mean(degrees)


GRAPH_METRICS = {
    "spectral_gap": spectral_gap,
    "mean_degree": mean_degree,
}
