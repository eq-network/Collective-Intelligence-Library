"""Pure delegation graph transformation."""
import jax.numpy as jnp
from typing import Callable

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from algebra.graph import GraphState

def create_delegation_transform() -> Callable[[GraphState], GraphState]:
    """Creates a transform that updates delegation edges from node attributes."""
    def transform(state: GraphState) -> GraphState:
     if "delegation_target" not in state.node_attrs:
         return state
     
     choices = state.node_attrs["delegation_target"]
     num_agents = choices.shape[0] # Get num_agents directly from the choices array

     if num_agents == 0:
         # Ensure an empty 'delegation' matrix is present if other adj_matrices exist,
         # or just return if adj_matrices itself might be None/empty.
         current_adj_matrices = state.adj_matrices if state.adj_matrices is not None else {}
         return state.replace(adj_matrices={
             **current_adj_matrices,
             "delegation": jnp.array([[]], dtype=jnp.float32).reshape(0,0) # Correct empty 2D array
         })

     delegation = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)
     for i in range(num_agents):
         choice = choices[i] # This is an int from jnp.array
         # Ensure choice is a scalar Python int for indexing if necessary, though JAX handles it.
         if choice >= 0 and choice < num_agents and choice != i:
             delegation = delegation.at[i, int(choice)].set(1.0)
     
     current_adj_matrices = state.adj_matrices if state.adj_matrices is not None else {}
     return state.replace(adj_matrices={
         **current_adj_matrices,
         "delegation": delegation
     })
    
    return transform