# transformations/message_passing.py
"""
A generic, pure transformation for executing a message-passing step on the graph.
"""
from typing import Callable, Dict, Any, List
import jax.numpy as jnp

from core.graph import GraphState
from core.category import Transform

# Type hints for clarity
Message = Dict[str, Any]
MessageGenerator = Callable[[GraphState, int], Message]  # (state, sender_id) -> message
MessageProcessor = Callable[[GraphState, int, List[Message]], Dict[str, Any]] # (state, receiver_id, messages) -> new_attrs

def create_message_passing_transform(
    connection_type: str,
    message_generator: MessageGenerator,
    message_processor: MessageProcessor
) -> Transform:
    """
    Creates a generic message-passing transformation.

    This is the core of the new simulation engine. It orchestrates the
    generation, transmission, and processing of messages between nodes.

    Args:
        connection_type: The key for the adjacency matrix that defines the communication topology.
        message_generator: A function that generates a message for a given sender node.
        message_processor: A function that updates a receiver node's state based on incoming messages.

    Returns:
        A pure graph-to-graph transformation function.
    """
    def transform(state: GraphState) -> GraphState:
        # 1. Setup
        active_indices = state.get_active_indices()
        active_mask = state.get_active_mask()
        adj_matrix = state.adj_matrices.get(connection_type)
        if adj_matrix is None:
            print(f"Warning: Connection type '{connection_type}' not found in adj_matrices. Skipping transform.")
            return state

        # 2. Generate messages (only from active nodes)
        messages = {}
        for i in active_indices:
            i_int = int(i)
            messages[i_int] = message_generator(state, i_int)

        # 3. Process messages for each active node
        new_node_attrs_updates: Dict[int, Dict[str, Any]] = {}
        for i in active_indices:
            i_int = int(i)

            # Find active nodes that send a message to node `i`
            sender_mask = (adj_matrix[:, i] > 0) & active_mask
            sender_indices = jnp.where(sender_mask)[0]

            # Collect incoming messages from active senders
            incoming_messages = [messages[int(j)] for j in sender_indices if int(j) in messages]

            # Process messages and get the attribute updates for node `i`
            updates = message_processor(state, i_int, incoming_messages)
            new_node_attrs_updates[i_int] = updates

        # 4. Apply all updates to create the new state
        # This approach ensures the transformation remains pure. All reads are from
        # the original `state`, and all writes are batched into the new state.
        final_node_attrs = state.node_attrs.copy()
        for attr_name in final_node_attrs.keys():
            # Check if any message processor returned an update for this attribute
            if any(attr_name in updates for updates in new_node_attrs_updates.values()):
                new_values = final_node_attrs[attr_name].copy()
                for i, updates in new_node_attrs_updates.items():
                    if attr_name in updates:
                        new_values = new_values.at[i].set(updates[attr_name])
                final_node_attrs[attr_name] = new_values

        return state.replace(node_attrs=final_node_attrs)
        
    return transform