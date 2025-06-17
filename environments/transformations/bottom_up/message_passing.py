"""
Pure message passing transformation for graph state.
"""
import jax.numpy as jnp
from typing import Callable

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from algebra.graph import GraphState
from algebra.property import attach_properties

def create_message_passing_transform(connection_type: str = "communication"):
    """
    Creates a pure message passing transformation.
    
    Message passing requires these attributes in state:
    - message_generator: Function in global_attrs that creates messages from node attributes
    - message_processor: Function in global_attrs that processes received messages
    - {connection_type}: Adjacency matrix defining the message flow graph
    """
    def transform(state: GraphState) -> GraphState:
        # Early return if required components missing
        if (connection_type not in state.adj_matrices or
           "message_generator" not in state.global_attrs or
           "message_processor" not in state.global_attrs):
            return state
            
        # Extract required components
        connections = state.adj_matrices[connection_type]
        generator = state.global_attrs["message_generator"]
        processor = state.global_attrs["message_processor"]
        num_nodes = state.num_nodes
        
        # Generate messages for each node
        messages = {}
        for i in range(num_nodes):
            node_attrs = {k: v[i] for k, v in state.node_attrs.items()}
            messages[i] = generator(i, node_attrs)
        
        # Distribute messages according to connection graph
        received = {i: [] for i in range(num_nodes)}
        for sender in range(num_nodes):
            for receiver in range(num_nodes):
                if connections[sender, receiver] > 0:
                    received[receiver].append(messages[sender])
        
        # Process messages and update node attributes
        new_attrs = {k: v.copy() for k, v in state.node_attrs.items()}
        for i in range(num_nodes):
            node_attrs = {k: v[i] for k, v in state.node_attrs.items()}
            updated = processor(i, node_attrs, received[i])
            
            # Update only existing attributes
            for attr, value in updated.items():
                if attr in new_attrs:
                    new_attrs[attr] = new_attrs[attr].at[i].set(value)
        
        return state.replace(node_attrs=new_attrs)
    
    return transform