"""
Graph representation as an immutable JAX-compatible structure.

This module defines the immutable graph state that forms the foundation
of our transformation system, designed for JAX compatibility.
"""
import jax
import jax.numpy as jnp
import dataclasses
from typing import Dict, Any, Set, Tuple, Optional, TypeVar, List, Callable
from functools import partial

@dataclasses.dataclass(frozen=True)
class GraphState:
    """
    Immutable JAX-compatible graph state representation.
    
    This class represents a complete graph state with node attributes,
    adjacency matrices, and global attributes. It's designed to be immutable
    and compatible with JAX transformations.
    """
    # Node attributes as arrays (each attribute is an array with shape [num_nodes, ...])
    node_attrs: Dict[str, jnp.ndarray]
    
    # Adjacency matrices (one per edge type, shape [num_nodes, num_nodes])
    adj_matrices: Dict[str, jnp.ndarray]
    
    # Global attributes (can include non-JAX objects)
    global_attrs: Dict[str, Any] = None
    
    def __post_init__(self):
        # Ensure global_attrs is initialized
        if self.global_attrs is None:
            object.__setattr__(self, 'global_attrs', {})
    
    def replace(self, **kwargs) -> 'GraphState':
        """
        Create a new graph state with updated components.
        
        This is the primary way to "modify" the graph, following
        JAX's immutability approach.
        """
        return dataclasses.replace(self, **kwargs)
    
    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        if not self.node_attrs:
            return 0
        return next(iter(self.node_attrs.values())).shape[0]
    
    def update_node_attrs(self, attr_name: str, new_values: jnp.ndarray) -> 'GraphState':
        """
        Create a new graph with an updated node attribute.
        """
        new_node_attrs = dict(self.node_attrs)
        new_node_attrs[attr_name] = new_values
        return self.replace(node_attrs=new_node_attrs)
    
    def update_adj_matrix(self, rel_name: str, new_matrix: jnp.ndarray) -> 'GraphState':
        """
        Create a new graph with an updated adjacency matrix.
        """
        new_adj_matrices = dict(self.adj_matrices)
        new_adj_matrices[rel_name] = new_matrix
        return self.replace(adj_matrices=new_adj_matrices)
    
    def update_global_attr(self, attr_name: str, value: Any) -> 'GraphState':
        """
        Create a new graph with an updated global attribute.
        """
        new_global_attrs = dict(self.global_attrs)
        new_global_attrs[attr_name] = value
        return self.replace(global_attrs=new_global_attrs)