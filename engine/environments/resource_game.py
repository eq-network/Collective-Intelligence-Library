"""
Resource game environment initialization.

This module provides setup functions for the resource game, where agents
trade resources, build trust, and interact with regulatory agents (banks, government).
"""
import jax.numpy as jnp
import jax.random as random
from typing import Tuple

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core import GraphState, initialize_graph_state


def create_resource_game(
    n_traders: int,
    n_resources: int,
    n_banks: int = 0,
    n_governments: int = 0,
    key: random.PRNGKey = None,
) -> GraphState:
    """
    Initialize a resource game with traders and optional regulatory agents.

    This creates a complete resource game state with:
    - Agents (traders, banks, governments) with initial resources
    - Connectivity structure (who can trade with whom)
    - Trust scores between agents
    - Empty message channels (for resource flows)

    Args:
        n_traders: Number of regular trading agents
        n_resources: Number of resource types (e.g., 5 different resources)
        n_banks: Number of bank agents (provide loans, liquidity)
        n_governments: Number of government agents (taxation, redistribution)
        key: JAX PRNGKey for random initialization. If None, uses PRNGKey(42)

    Returns:
        GraphState initialized for resource game with:
            - node_types: 0=trader, 1=bank, 2=government
            - node_attrs["resources"]: initial resource stockpiles
            - edge_attrs["messages"]: zeros (no messages yet)
            - edge_attrs["trust"]: 0.5 (neutral trust initially)
            - adj_matrices["connections"]: connectivity structure

    Example:
        # Simple: 10 traders, 5 resources
        key = random.PRNGKey(42)
        state = create_resource_game(n_traders=10, n_resources=5, key=key)

        # With regulatory agents
        state = create_resource_game(
            n_traders=20,
            n_resources=5,
            n_banks=1,
            n_governments=1,
            key=key
        )
    """
    if key is None:
        key = random.PRNGKey(42)

    # Total number of agents
    n_agents = n_traders + n_banks + n_governments

    # Generate initial resources
    key, subkey = random.split(key)
    initial_resources = _generate_initial_resources(
        n_traders, n_banks, n_governments, n_resources, subkey
    )

    # Initialize base graph state
    state = initialize_graph_state(
        n_agents=n_agents,
        n_resources=n_resources,
        initial_resources=initial_resources
    )

    # Set node types: 0=trader, 1=bank, 2=government
    node_types = jnp.concatenate([
        jnp.zeros(n_traders, dtype=jnp.int32),      # Traders
        jnp.ones(n_banks, dtype=jnp.int32),         # Banks
        jnp.ones(n_governments, dtype=jnp.int32) * 2  # Governments
    ])
    state = state.replace(node_types=node_types)

    # Add connectivity structure
    connections = _create_network_topology(n_traders, n_banks, n_governments)
    state = state.update_adj_matrix("connections", connections)

    # Add trust scores (initially neutral: 0.5)
    trust = jnp.ones((n_agents, n_agents)) * 0.5
    state = state.update_edge_attrs("trust", trust)

    # Add message channels (initially empty)
    messages = jnp.zeros((n_agents, n_agents, n_resources))
    state = state.update_edge_attrs("messages", messages)

    return state


def _generate_initial_resources(
    n_traders: int,
    n_banks: int,
    n_governments: int,
    n_resources: int,
    key: random.PRNGKey
) -> jnp.ndarray:
    """
    Generate initial resource allocations for all agents.

    Traders: Random resources between 5-15
    Banks: Large stockpiles (100 per resource) - liquidity providers
    Governments: Medium stockpiles (50 per resource) - for redistribution
    """
    # Traders: random resources
    trader_resources = random.uniform(
        key,
        (n_traders, n_resources),
        minval=5.0,
        maxval=15.0
    )

    # Banks: large stockpiles
    bank_resources = jnp.ones((n_banks, n_resources)) * 100.0

    # Governments: medium stockpiles
    gov_resources = jnp.ones((n_governments, n_resources)) * 50.0

    # Concatenate all
    all_resources = jnp.concatenate([
        trader_resources,
        bank_resources,
        gov_resources
    ], axis=0)

    return all_resources


def _create_network_topology(
    n_traders: int,
    n_banks: int,
    n_governments: int
) -> jnp.ndarray:
    """
    Create connectivity structure for the resource game.

    Topology:
    - Traders connected in a ring (can trade with neighbors)
    - Banks connected to all traders (provide liquidity to everyone)
    - Governments connected to all traders (can tax/redistribute to everyone)
    - Banks and governments not directly connected to each other

    Returns:
        Adjacency matrix (n_agents, n_agents) where 1 = connected, 0 = not connected
    """
    n_agents = n_traders + n_banks + n_governments
    connections = jnp.zeros((n_agents, n_agents))

    # Traders in a ring: each connected to left and right neighbor
    for i in range(n_traders):
        left = (i - 1) % n_traders
        right = (i + 1) % n_traders
        connections = connections.at[i, left].set(1.0)
        connections = connections.at[i, right].set(1.0)

    # Banks connected to all traders
    for bank_idx in range(n_traders, n_traders + n_banks):
        for trader_idx in range(n_traders):
            connections = connections.at[bank_idx, trader_idx].set(1.0)
            connections = connections.at[trader_idx, bank_idx].set(1.0)

    # Governments connected to all traders
    for gov_idx in range(n_traders + n_banks, n_agents):
        for trader_idx in range(n_traders):
            connections = connections.at[gov_idx, trader_idx].set(1.0)
            connections = connections.at[trader_idx, gov_idx].set(1.0)

    return connections
