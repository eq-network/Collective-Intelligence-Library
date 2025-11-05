"""
Observation builders for the Farmer's Market environment.

These functions package GraphState information into observations that agents can use.
The observation determines what information each agent can "see" from the global state.
"""
import jax.numpy as jnp
from typing import Dict, Any, List
from core.graph import GraphState


def build_farmer_observation(
    state: GraphState,
    agent_id: int,
    include_neighbors: bool = True,
    include_global: bool = True
) -> Dict[str, Any]:
    """
    Build an observation for a single farmer agent.

    Args:
        state: Current graph state
        agent_id: ID of the agent to build observation for
        include_neighbors: Whether to include neighbor information
        include_global: Whether to include global attributes

    Returns:
        Observation dictionary containing:
        - agent_id: The agent's ID
        - round: Current round number
        - my_resources: Dict of agent's current resources
        - my_growth_rates: Dict of agent's growth rates
        - neighbors: List of neighbor IDs
        - neighbor_resources: Dict[neighbor_id → resources] (if include_neighbors)
        - resource_types: List of all resource types
        - network_size: Total number of agents (if include_global)
    """
    observation = {
        "agent_id": agent_id,
        "round": state.global_attrs.get("round", 0),
    }

    # Get agent's own resources
    resource_types = state.global_attrs.get("resource_types", [])
    observation["resource_types"] = resource_types

    my_resources = {}
    my_growth_rates = {}

    for resource_type in resource_types:
        resource_key = f"resources_{resource_type}"
        growth_key = f"growth_rate_{resource_type}"

        if resource_key in state.node_attrs:
            my_resources[resource_type] = float(state.node_attrs[resource_key][agent_id])

        if growth_key in state.node_attrs:
            my_growth_rates[resource_type] = float(state.node_attrs[growth_key][agent_id])

    observation["my_resources"] = my_resources
    observation["my_growth_rates"] = my_growth_rates

    # Get neighbors from trade network
    if "trade_network" in state.adj_matrices:
        trade_network = state.adj_matrices["trade_network"]
        # Find neighbors (where connection exists)
        neighbor_mask = trade_network[agent_id, :] > 0
        neighbor_ids = jnp.where(neighbor_mask)[0]
        observation["neighbors"] = [int(nid) for nid in neighbor_ids]

        # Include neighbor resources if requested
        if include_neighbors:
            neighbor_resources = {}
            for nid in neighbor_ids:
                neighbor_res = {}
                for resource_type in resource_types:
                    resource_key = f"resources_{resource_type}"
                    if resource_key in state.node_attrs:
                        neighbor_res[resource_type] = float(state.node_attrs[resource_key][int(nid)])
                neighbor_resources[int(nid)] = neighbor_res

            observation["neighbor_resources"] = neighbor_resources
    else:
        observation["neighbors"] = []
        observation["neighbor_resources"] = {}

    # Global information
    if include_global:
        observation["network_size"] = state.num_nodes
        observation["total_trades"] = state.global_attrs.get("total_trades", 0)

    return observation


def build_all_observations(
    state: GraphState,
    include_neighbors: bool = True,
    include_global: bool = True
) -> List[Dict[str, Any]]:
    """
    Build observations for all agents in the state.

    Args:
        state: Current graph state
        include_neighbors: Whether to include neighbor information
        include_global: Whether to include global attributes

    Returns:
        List of observation dicts, one per agent
    """
    return [
        build_farmer_observation(state, agent_id, include_neighbors, include_global)
        for agent_id in range(state.num_nodes)
    ]


def build_limited_observation(
    state: GraphState,
    agent_id: int,
    observation_radius: int = 1
) -> Dict[str, Any]:
    """
    Build a limited observation that only includes information within a certain radius.

    Useful for testing information propagation and local coordination.

    Args:
        state: Current graph state
        agent_id: ID of the agent
        observation_radius: How many hops away the agent can observe

    Returns:
        Observation with limited information
    """
    obs = build_farmer_observation(state, agent_id, include_neighbors=False, include_global=False)

    # Only include immediate neighbors (radius 1)
    if observation_radius >= 1 and "trade_network" in state.adj_matrices:
        trade_network = state.adj_matrices["trade_network"]
        neighbor_mask = trade_network[agent_id, :] > 0
        neighbor_ids = jnp.where(neighbor_mask)[0]
        obs["neighbors"] = [int(nid) for nid in neighbor_ids]

        # For radius > 1, would need to do BFS/DFS on network
        # Not implemented yet

    return obs
