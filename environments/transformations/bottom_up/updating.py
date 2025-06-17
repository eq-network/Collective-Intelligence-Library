from typing import TypeVar, Generic, Callable, List, Set, Dict, Any

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from algebra.graph import GraphState

# transformations/bottom_up/updating.py
def belief_update_transform(
    state: GraphState,
    update_function: Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]
) -> GraphState:
    """
    Pure transformation for updating node beliefs based on neighbor information.
    
    Args:
        state: Current graph state
        update_function: Function that computes new beliefs based on current 
                         beliefs and received messages
    
    Returns:
        Updated GraphState
    """
    # Extract relevant state
    beliefs = state.node_attrs.get("belief")
    connections = state.adj_matrices.get("communication")
    messages = state.node_attrs.get("message")
    
    if beliefs is None or connections is None or messages is None:
        return state
    
    num_nodes = state.num_nodes
    new_beliefs = beliefs.copy()
    
    # For each node, collect neighbor messages and update beliefs
    for i in range(num_nodes):
        # Extract current beliefs
        node_beliefs = {k: v[i] for k, v in beliefs.items()}
        
        # Collect neighbor messages
        neighbor_messages = []
        for j in range(num_nodes):
            if connections[j, i] > 0:  # j connected to i
                neighbor_message = {k: v[j] for k, v in messages.items()}
                neighbor_messages.append(neighbor_message)
        
        # Apply update function
        updated_beliefs = update_function(node_beliefs, neighbor_messages)
        
        # Update belief state
        for k, v in updated_beliefs.items():
            if k in new_beliefs:
                new_beliefs[k] = new_beliefs[k].at[i].set(v)
    
    return state.update_node_attrs("belief", new_beliefs)