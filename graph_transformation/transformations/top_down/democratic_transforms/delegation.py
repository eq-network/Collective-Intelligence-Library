"""Pure delegation graph transformation."""
import jax.numpy as jnp
from typing import Callable

from core.graph import GraphState

def create_delegation_transform() -> Callable[[GraphState], GraphState]:
    """Creates a transform that updates delegation edges from node attributes."""
    def transform(state: GraphState) -> GraphState:
        if "delegation_choices" not in state.node_attrs:
            return state
        
        choices = state.node_attrs["delegation_choices"]
        num_agents = state.num_nodes
        
        # Convert choice vector to delegation matrix
        delegation = jnp.zeros((num_agents, num_agents))
        for i in range(num_agents):
            choice = choices[i]
            if choice >= 0 and choice < num_agents and choice != i:
                delegation = delegation.at[i, int(choice)].set(1.0)
        
        # Update adjacency matrix
        return state.replace(adj_matrices={
            **state.adj_matrices,
            "delegation": delegation
        })
    
    return transform