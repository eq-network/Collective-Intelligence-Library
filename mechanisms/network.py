"""
Network transforms: information diffusion and belief propagation.

Implements network-based diffusion processes as pure graph transformations.
"""

import jax.numpy as jnp
from typing import Dict, Any
from core.graph import GraphState
from core.category import Transform
from core.mask import apply_masked_update


def _ensure_row_stochastic(W: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize rows to sum to 1 (make row-stochastic)."""
    rs = W.sum(axis=1, keepdims=True)
    rs = jnp.where(rs < eps, 1.0, rs)
    return W / rs


def create_belief_diffusion_transform(
    network_name: str = "social",
    belief_attr: str = "belief",
    diffusion_rate: float = 0.3,
) -> Transform:
    """
    Create a belief diffusion transform using network averaging.

    Each agent's belief is updated as:
        belief_new = (1 - rate) * belief_old + rate * avg_neighbor_beliefs

    Args:
        network_name: Key in adj_matrices to use for diffusion
        belief_attr: Node attribute containing beliefs to diffuse
        diffusion_rate: Rate of belief update (0 = no change, 1 = full averaging)

    Returns:
        Transform that applies belief diffusion
    """

    def diffusion_transform(state: GraphState) -> GraphState:
        # Get network and beliefs
        W = jnp.asarray(state.adj_matrices[network_name])
        x = jnp.asarray(state.node_attrs[belief_attr])

        # Normalize network to row-stochastic (weighted average)
        Wn = _ensure_row_stochastic(W)

        # Apply diffusion: convex combination of self and neighbors
        y = (1.0 - diffusion_rate) * x + diffusion_rate * (Wn @ x)

        # Apply mask if present (only update active agents)
        mask = jnp.asarray(state.global_attrs.get("active_mask", jnp.ones_like(x)))
        y_masked = apply_masked_update(x, y, mask, "node")

        return state.update_node_attrs(belief_attr, y_masked)

    return diffusion_transform


def create_information_spread_transform(
    network_name: str = "social",
    source_attr: str = "belief",
    target_attr: str = "belief",
    damping: float = 0.9,
) -> Transform:
    """
    Create an information spread transform with damping.

    Similar to belief diffusion but emphasizes neighbor influence:
        target_new = damping * avg_neighbor_source + (1 - damping) * source_old

    Args:
        network_name: Key in adj_matrices to use for spreading
        source_attr: Source node attribute to spread
        target_attr: Target node attribute to update (can be same as source)
        damping: Weight on neighbor influence (0 = no spread, 1 = full replacement)

    Returns:
        Transform that applies information spreading
    """

    def spread_transform(state: GraphState) -> GraphState:
        # Get network and source values
        W = jnp.asarray(state.adj_matrices[network_name])
        x = jnp.asarray(state.node_attrs[source_attr])

        # Normalize network
        Wn = _ensure_row_stochastic(W)

        # Apply spread: weighted combination
        y = damping * (Wn @ x) + (1.0 - damping) * x

        return state.update_node_attrs(target_attr, y)

    return spread_transform


def create_opinion_dynamics_transform(
    network_name: str = "social",
    opinion_attr: str = "opinion",
    confidence_threshold: float = 0.5,
    convergence_rate: float = 0.3,
) -> Transform:
    """
    Create a bounded confidence opinion dynamics transform (Deffuant model).

    Agents only influence each other if their opinions are within confidence_threshold.

    Args:
        network_name: Key in adj_matrices for social network
        opinion_attr: Node attribute with continuous opinions
        confidence_threshold: Maximum opinion distance for influence
        convergence_rate: Rate of opinion convergence toward neighbors

    Returns:
        Transform implementing bounded confidence dynamics
    """

    def opinion_dynamics_transform(state: GraphState) -> GraphState:
        W = jnp.asarray(state.adj_matrices[network_name])
        opinions = jnp.asarray(state.node_attrs[opinion_attr])

        # Compute pairwise opinion distances
        diff = opinions[:, jnp.newaxis] - opinions[jnp.newaxis, :]

        # Only consider neighbors within confidence threshold
        within_threshold = jnp.abs(diff) <= confidence_threshold
        W_filtered = W * within_threshold.astype(float)

        # Normalize filtered network
        Wn = _ensure_row_stochastic(W_filtered)

        # Update opinions toward compatible neighbors
        avg_neighbor_opinion = Wn @ opinions
        new_opinions = (
            1.0 - convergence_rate
        ) * opinions + convergence_rate * avg_neighbor_opinion

        return state.update_node_attrs(opinion_attr, new_opinions)

    return opinion_dynamics_transform


def create_cascade_transform(
    network_name: str = "social",
    activation_attr: str = "activated",
    threshold: float = 0.3,
) -> Transform:
    """
    Create a threshold cascade transform (information/behavior spreading).

    Agents activate if fraction of activated neighbors exceeds threshold.

    Args:
        network_name: Key in adj_matrices for influence network
        activation_attr: Binary node attribute (0 = inactive, 1 = active)
        threshold: Fraction of activated neighbors required for activation

    Returns:
        Transform implementing threshold cascades
    """

    def cascade_transform(state: GraphState) -> GraphState:
        W = jnp.asarray(state.adj_matrices[network_name])
        activated = jnp.asarray(state.node_attrs[activation_attr])

        # Normalize network to get neighbor fractions
        Wn = _ensure_row_stochastic(W)

        # Compute fraction of activated neighbors
        neighbor_activation = Wn @ activated

        # Activate if threshold exceeded (irreversible)
        new_activated = jnp.maximum(
            activated, (neighbor_activation >= threshold).astype(float)
        )

        return state.update_node_attrs(activation_attr, new_activated)

    return cascade_transform


def create_pagerank_transform(
    network_name: str = "social",
    rank_attr: str = "pagerank",
    damping: float = 0.85,
    num_iterations: int = 10,
) -> Transform:
    """
    Create a PageRank transform to compute node centrality.

    Implements the PageRank algorithm on the network.

    Args:
        network_name: Key in adj_matrices for link network
        rank_attr: Node attribute to store PageRank scores
        damping: Damping factor (typically 0.85)
        num_iterations: Number of power iterations

    Returns:
        Transform that computes PageRank
    """

    def pagerank_transform(state: GraphState) -> GraphState:
        W = jnp.asarray(state.adj_matrices[network_name])
        n = W.shape[0]

        # Column-normalize for PageRank (transpose for row operations)
        W_col_norm = _ensure_row_stochastic(W.T).T

        # Initialize uniform rank
        rank = jnp.ones(n) / n

        # Power iteration
        teleport = (1.0 - damping) / n
        for _ in range(num_iterations):
            rank = damping * (W_col_norm @ rank) + teleport

        return state.update_node_attrs(rank_attr, rank)

    return pagerank_transform


def apply_masked_update(old, new, mask, field_kind: str):
    m = mask.astype(new.dtype)
    if field_kind == "node":  # [N]
        return old + (new - old) * m
    elif field_kind == "row_matrix":  # [N,N], gate by row
        return old + (new - old) * m[:, None]
    elif field_kind == "both_matrix":  # [N,N], require both endpoints
        return old + (new - old) * (m[:, None] * m[None, :])
    else:
        raise ValueError("unknown field_kind")
