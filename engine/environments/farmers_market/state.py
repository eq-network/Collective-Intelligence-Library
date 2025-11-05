"""
State initialization for the Farmer's Market environment.

Defines how to create the initial GraphState with farmers, resources, and networks.
"""
import jax.numpy as jnp
import jax.random as jr
from typing import List, Dict, Optional
from core.graph import GraphState


def create_farmers_market_state(
    num_farmers: int,
    resource_types: List[str],
    initial_resources_per_farmer: Dict[str, float],
    network_density: float = 0.3,
    seed: int = 42
) -> GraphState:
    """
    Create the initial state for a farmer's market simulation.

    Args:
        num_farmers: Number of farmers in the market
        resource_types: List of resource names (e.g., ["apples", "wheat", "corn"])
        initial_resources_per_farmer: Initial amount of each resource per farmer
        network_density: Probability of trade connection between any two farmers
        seed: Random seed for reproducibility

    Returns:
        GraphState with farmers, resources, and trade network initialized

    Example:
        >>> state = create_farmers_market_state(
        ...     num_farmers=10,
        ...     resource_types=["apples", "wheat", "corn"],
        ...     initial_resources_per_farmer={"apples": 100, "wheat": 100, "corn": 100},
        ...     network_density=0.3,
        ...     seed=42
        ... )
    """
    key = jr.PRNGKey(seed)

    # Initialize node attributes: each farmer has resources
    node_attrs = {}

    # Each resource type gets its own node attribute array
    for resource_type in resource_types:
        initial_amount = initial_resources_per_farmer.get(resource_type, 0.0)
        # Add some random variation (±20%)
        key, subkey = jr.split(key)
        variation = jr.uniform(subkey, shape=(num_farmers,), minval=0.8, maxval=1.2)
        node_attrs[f"resources_{resource_type}"] = jnp.array(
            [initial_amount * v for v in variation]
        )

    # Growth rates: how fast each farmer grows each resource
    # This could be specialized per farmer (some better at apples, others at wheat)
    for resource_type in resource_types:
        key, subkey = jr.split(key)
        # Growth rates between 1.05 and 1.15 (5-15% growth per round)
        growth_rates = jr.uniform(subkey, shape=(num_farmers,), minval=1.05, maxval=1.15)
        node_attrs[f"growth_rate_{resource_type}"] = growth_rates

    # Create trade network: random graph with given density
    key, subkey = jr.split(key)
    trade_connections = jr.bernoulli(
        subkey,
        p=network_density,
        shape=(num_farmers, num_farmers)
    ).astype(float)

    # Remove self-connections
    trade_connections = trade_connections.at[jnp.diag_indices(num_farmers)].set(0)

    # Make network symmetric (undirected trading relationships)
    trade_connections = jnp.maximum(trade_connections, trade_connections.T)

    # Node types: all are farmers (type 0)
    node_types = jnp.zeros(num_farmers, dtype=int)

    # Global attributes
    global_attrs = {
        "round": 0,
        "num_farmers": num_farmers,
        "resource_types": resource_types,
        "seed": seed,
        "total_trades": 0,
        "history": []
    }

    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={"trade_network": trade_connections},
        global_attrs=global_attrs
    )


def create_simple_farmers_market(
    num_farmers: int = 10,
    seed: int = 42
) -> GraphState:
    """
    Convenience function to create a simple farmer's market with default settings.

    Creates a market with three resource types: apples, wheat, and corn.

    Args:
        num_farmers: Number of farmers
        seed: Random seed

    Returns:
        Initialized GraphState
    """
    return create_farmers_market_state(
        num_farmers=num_farmers,
        resource_types=["apples", "wheat", "corn"],
        initial_resources_per_farmer={
            "apples": 100.0,
            "wheat": 100.0,
            "corn": 100.0
        },
        network_density=0.3,
        seed=seed
    )
