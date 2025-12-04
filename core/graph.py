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

class CapacityExceededError(Exception):
    """Raised when adding node beyond capacity."""
    pass


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
        capacity: Optional[int] - Maximum nodes (None = backward compatible dynamic mode)

    Matrix Semantics:
        - node_attrs represent diagonal blocks: state[i, i] = agent i's memory
        - adj_matrices represent off-diagonal: state[i, j] = message from i to j

    Capacity Mode (Optional):
        - When capacity is set, arrays are padded to fixed size
        - Inactive nodes marked with node_type=-1
        - O(1) add/remove operations by slot activation
        - Use get_active_indices() to filter to active nodes

    Performance:
        - Separate matrices (one per resource) allows selective updates
        - JAX operations work efficiently on individual matrices
        - Use get_node_state(i) for fast access to all of node i's attributes
        - Capacity mode enables JIT compilation with fixed shapes
    """
    node_types: jnp.ndarray

    node_attrs: Dict[str, jnp.ndarray]
    adj_matrices: Dict[str, jnp.ndarray]
    global_attrs: Dict[str, Any] = field(default_factory=dict)

    # Optional capacity (None = backward compatible dynamic mode)
    capacity: Optional[int] = None
    
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
        """Get number of ACTIVE nodes."""
        if self.capacity is None:
            # Backward compatible: all nodes active
            if not self.node_attrs:
                return 0
            return next(iter(self.node_attrs.values())).shape[0]
        # Capacity mode: count active nodes
        return int(jnp.sum(self.node_types != -1))

    @property
    def is_capacity_mode(self) -> bool:
        """Check if using fixed-capacity mode."""
        return self.capacity is not None

    def get_active_indices(self) -> jnp.ndarray:
        """Get indices of active nodes."""
        if not self.is_capacity_mode:
            return jnp.arange(self.num_nodes)
        return jnp.where(self.node_types != -1)[0]

    def get_active_mask(self) -> jnp.ndarray:
        """Get boolean mask for active nodes."""
        if not self.is_capacity_mode:
            return jnp.ones(self.num_nodes, dtype=bool)
        return self.node_types != -1
    
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


def create_padded_state(
    capacity: int,
    initial_active: int,
    node_types_init: Optional[jnp.ndarray] = None,
    node_attrs_init: Optional[Dict[str, jnp.ndarray]] = None,
    adj_matrices_init: Optional[Dict[str, jnp.ndarray]] = None,
    global_attrs: Optional[Dict[str, Any]] = None
) -> GraphState:
    """
    Create GraphState with fixed capacity.

    Args:
        capacity: Maximum nodes (fixed array size)
        initial_active: Number of initially active nodes
        node_types_init: Initial types [initial_active]
        node_attrs_init: {attr_name: [initial_active]}
        adj_matrices_init: {rel_name: [initial_active, initial_active]}
        global_attrs: Global attributes dict

    Returns:
        GraphState with padded arrays

    Example:
        >>> state = create_padded_state(
        ...     capacity=10,
        ...     initial_active=3,
        ...     node_attrs_init={"resources": jnp.array([100, 100, 100])},
        ...     adj_matrices_init={"network": jnp.eye(3)}
        ... )
        >>> state.num_nodes  # Returns 3 (active nodes)
        >>> state.capacity   # Returns 10 (total capacity)
    """
    # Validate
    if initial_active > capacity:
        raise ValueError(f"initial_active ({initial_active}) exceeds capacity ({capacity})")

    # Pad node_types: [active types..., -1, -1, ...]
    active_types = node_types_init if node_types_init is not None else jnp.zeros(initial_active, dtype=jnp.int32)
    inactive_types = jnp.full(capacity - initial_active, -1, dtype=jnp.int32)
    node_types = jnp.concatenate([active_types, inactive_types])

    # Pad node_attrs: each becomes [capacity] with zeros for inactive
    node_attrs = {}
    for attr_name, active_values in (node_attrs_init or {}).items():
        padding = jnp.zeros(capacity - initial_active, dtype=active_values.dtype)
        node_attrs[attr_name] = jnp.concatenate([active_values, padding])

    # Pad adj_matrices: each becomes [capacity, capacity]
    adj_matrices = {}
    for rel_name, active_matrix in (adj_matrices_init or {}).items():
        padded = jnp.zeros((capacity, capacity), dtype=active_matrix.dtype)
        padded = padded.at[:initial_active, :initial_active].set(active_matrix)
        adj_matrices[rel_name] = padded

    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices=adj_matrices,
        global_attrs=global_attrs or {},
        capacity=capacity
    )