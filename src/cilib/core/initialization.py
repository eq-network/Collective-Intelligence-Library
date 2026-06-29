"""
Graph state initialization utilities.

This module provides functions for creating initial GraphState instances
with minimal required parameters. No magic defaults - all inputs must be explicit.
"""
import jax.numpy as jnp
from typing import Optional, Dict, Any

from .graph import GraphState


def initialize_graph_state(
    n_agents: int,
    n_resources: int,
    initial_resources: jnp.ndarray,
) -> GraphState:
    """
    Initialize a GraphState with agents and resources.

    This is the minimal initialization - just agents with resource stockpiles.
    Everything else (connectivity, edge data, memory, etc.) can be added later
    using the GraphState update methods.

    Args:
        n_agents: Number of agents in the system (must be > 0)
        n_resources: Number of resource types (must be > 0)
        initial_resources: JAX array of shape (n_agents, n_resources) containing
                          initial resource values for each agent.
                          Generate random values with jax.random if desired.

    Returns:
        GraphState with:
            - node_types: all zeros (generic agents)
            - node_attrs: {"resources": initial_resources}
            - edge_attrs: {} (empty - add later if needed)
            - adj_matrices: {} (empty - add later if needed)
            - global_attrs: {"round": 0}

    Raises:
        AssertionError: If inputs are invalid or shapes don't match

    Examples:
        # Deterministic initialization
        resources = jnp.ones((10, 5)) * 10.0
        state = initialize_graph_state(10, 5, resources)

        # Random initialization (explicit)
        import jax.random as random
        key = random.PRNGKey(42)
        resources = random.uniform(key, (10, 5), minval=5, maxval=15)
        state = initialize_graph_state(10, 5, resources)

        # Add connectivity later
        connections = jnp.eye(10)
        state = state.update_adj_matrix("connections", connections)

        # Add edge data later
        trust = jnp.ones((10, 10)) * 0.5
        state = state.update_edge_attrs("trust", trust)
    """
    # Validate inputs
    assert n_agents > 0, "n_agents must be positive"
    assert n_resources > 0, "n_resources must be positive"
    assert initial_resources.shape == (n_agents, n_resources), \
        f"Resources shape mismatch: expected ({n_agents}, {n_resources}), got {initial_resources.shape}"
    assert jnp.all(jnp.isfinite(initial_resources)), \
        "initial_resources must contain finite values (no NaN or inf)"

    # Create node types (all 0 = generic agents by default)
    node_types = jnp.zeros(n_agents, dtype=jnp.int32)

    return GraphState(
        node_types=node_types,
        node_attrs={
            "resources": initial_resources,
        },
        edge_attrs={},
        adj_matrices={},
        global_attrs={
            "round": 0,
        }
    )
