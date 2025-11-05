"""
Graph representation as an immutable JAX-compatible structure.

GraphState IS the matrix representation. The entire system is built on matrix operations:

Matrix Structure:
- node_attrs (diagonal blocks): Agent i's local memory/state (self-loops, memory retention)
- adj_matrices (off-diagonal): Message passing from agent i to agent j

Transform Types:
1. Global transforms: Operate on entire matrices (e.g., apply discount to all resources)
2. Local transforms: Operate on specific nodes/edges (e.g., update node i's state)
3. Message passing: Update off-diagonal entries (i→j communication)
4. Memory retention: Update diagonal entries (i→i state persistence)

Example:
    # Memory retention (diagonal)
    state = state.update_node_attrs("resources", new_resources)

    # Message passing (off-diagonal)
    state = state.update_adj_matrix("trade_network", new_network)

    # Get all attributes for a specific node
    node_state = state.get_node_state(agent_id)
"""
import jax
import jax.numpy as jnp
import dataclasses
from typing import Dict, Any, Set, Tuple, Optional, TypeVar, List, Callable
from functools import partial
from dataclasses import field

@dataclasses.dataclass(frozen=True)
class GraphState:
    """
    Immutable JAX-compatible matrix-based state representation.

    GraphState IS the matrix. All operations are matrix transformations.

    Structure:
        node_types: [N] - Type labels for each agent
        node_attrs: {attr_name: [N]} - Diagonal matrices (agent i's local state)
        adj_matrices: {rel_name: [N, N]} - Off-diagonal matrices (i→j messages)
        global_attrs: {key: value} - Environment-level metadata

    Matrix Semantics:
        - node_attrs represent diagonal blocks: state[i, i] = agent i's memory
        - adj_matrices represent off-diagonal: state[i, j] = message from i to j

    Performance:
        - Separate matrices (one per resource) allows selective updates
        - JAX operations work efficiently on individual matrices
        - Use get_node_state(i) for fast access to all of node i's attributes
    """
    node_types: jnp.ndarray

    node_attrs: Dict[str, jnp.ndarray]
    adj_matrices: Dict[str, jnp.ndarray]
    global_attrs: Dict[str, Any] = field(default_factory=dict)
    
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

    def get_node_state(self, node_id: int) -> Dict[str, Any]:
        """
        Get all attributes for a specific node (fast node-level access).

        This efficiently gathers all of node i's state across separate matrices.

        Args:
            node_id: Index of node to query

        Returns:
            Dict with all node attributes and outgoing edges:
            {
                "type": node_type,
                "attrs": {attr_name: value, ...},
                "edges": {rel_name: {target_id: weight, ...}, ...}
            }

        Example:
            state.get_node_state(0)
            # {"type": 0, "attrs": {"resources_apples": 100, ...}, "edges": {"trade": {1: 1.0}}}
        """
        if node_id < 0 or node_id >= self.num_nodes:
            raise ValueError(f"Invalid node_id: {node_id}")

        # Gather node attributes (diagonal entries)
        attrs = {}
        for attr_name, values in self.node_attrs.items():
            attrs[attr_name] = float(values[node_id])

        # Gather outgoing edges (row in adjacency matrices)
        edges = {}
        for rel_name, matrix in self.adj_matrices.items():
            row = matrix[node_id]
            # Only include non-zero edges
            edges[rel_name] = {
                j: float(row[j])
                for j in range(self.num_nodes)
                if row[j] != 0 and j != node_id
            }

        return {
            "type": int(self.node_types[node_id]),
            "attrs": attrs,
            "edges": edges
        }

    def update_node_state(self, node_id: int, attr_updates: Dict[str, float]) -> 'GraphState':
        """
        Update multiple attributes for a specific node at once.

        This is more efficient than multiple update_node_attrs calls.

        Args:
            node_id: Index of node to update
            attr_updates: Dict of {attr_name: new_value}

        Returns:
            New GraphState with updated node attributes

        Example:
            state = state.update_node_state(0, {
                "resources_apples": 150,
                "resources_wheat": 200
            })
        """
        if node_id < 0 or node_id >= self.num_nodes:
            raise ValueError(f"Invalid node_id: {node_id}")

        new_node_attrs = dict(self.node_attrs)

        for attr_name, new_value in attr_updates.items():
            if attr_name not in new_node_attrs:
                raise ValueError(f"Unknown attribute: {attr_name}")

            # Update single entry in array
            new_node_attrs[attr_name] = new_node_attrs[attr_name].at[node_id].set(new_value)

        return self.replace(node_attrs=new_node_attrs)