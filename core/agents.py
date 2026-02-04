"""
Agent interface: observation → action on GraphState.
"""
from typing import Protocol, TypeAlias
import jax.numpy as jnp
from jax import random

from .graph import GraphState

ObservationMatrix: TypeAlias = jnp.ndarray
ActionMatrix: TypeAlias = jnp.ndarray


class Policy(Protocol):
    """Policy: observation matrix → action matrix."""
    def __call__(self, obs: ObservationMatrix, key: random.PRNGKey) -> ActionMatrix:
        ...


def get_observation(state: GraphState, agent_id: int) -> ObservationMatrix:
    """
    Extract observation matrix for agent from GraphState.

    Samples from: node attributes + adjacency + edge attributes.
    """
    n_agents = len(state.node_types)

    if state.adj_matrices:
        adjacency = list(state.adj_matrices.values())[0][agent_id]
    else:
        adjacency = jnp.ones(n_agents).at[agent_id].set(0)

    my_attrs = jnp.concatenate([
        jnp.atleast_1d(attr[agent_id]).flatten()
        for attr in state.node_attrs.values()
    ])

    neighbor_attrs = jnp.concatenate([
        attr.flatten() for attr in state.node_attrs.values()
    ])

    if state.edge_attrs:
        edge_data = jnp.concatenate([
            attr[agent_id].flatten() if attr.ndim > 1 else jnp.array([attr[agent_id]])
            for attr in state.edge_attrs.values()
        ])
    else:
        edge_data = jnp.array([])

    return jnp.concatenate([my_attrs, neighbor_attrs, edge_data, adjacency])


def apply_action(state: GraphState, agent_id: int, action: ActionMatrix) -> GraphState:
    """
    Apply agent action to GraphState.

    Action updates edge attributes (messages sent).
    """
    n_agents = len(state.node_types)
    n_resources = state.node_attrs["resources"].shape[1]

    transfers = action.reshape(n_agents, n_resources)

    new_resources = state.node_attrs["resources"]
    new_resources = new_resources.at[agent_id].add(-jnp.sum(transfers, axis=0))
    new_resources = new_resources.at[:].add(transfers)
    new_resources = jnp.maximum(new_resources, 0.0)

    new_state = state.update_node_attrs("resources", new_resources)

    if "messages" in state.edge_attrs:
        new_messages = state.edge_attrs["messages"].at[agent_id].set(transfers)
        new_state = new_state.update_edge_attrs("messages", new_messages)

    return new_state
