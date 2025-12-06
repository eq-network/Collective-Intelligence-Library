"""
Belief update transformation for updating node beliefs based on neighbor information.
"""
from typing import Callable, List, Dict, Any

from core.graph import GraphState

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

    active_indices = state.get_active_indices()
    active_mask = state.get_active_mask()
    new_beliefs = beliefs.copy()

    # For each active node, collect neighbor messages and update beliefs
    for i in active_indices:
        i_int = int(i)
        # Extract current beliefs
        node_beliefs = {k: v[i_int] for k, v in beliefs.items()}

        # Collect neighbor messages from active nodes
        neighbor_messages = []
        for j in active_indices:
            j_int = int(j)
            if connections[j_int, i_int] > 0:  # j connected to i
                neighbor_message = {k: v[j_int] for k, v in messages.items()}
                neighbor_messages.append(neighbor_message)

        # Apply update function
        updated_beliefs = update_function(node_beliefs, neighbor_messages)

        # Update belief state
        for k, v in updated_beliefs.items():
            if k in new_beliefs:
                new_beliefs[k] = new_beliefs[k].at[i_int].set(v)

    return state.update_node_attrs("belief", new_beliefs)