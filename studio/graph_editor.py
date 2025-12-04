"""
Graph editing operations.

Pure functions for adding/removing nodes and edges from GraphState.

Supports both capacity mode (O(1) operations via slot activation) and
dynamic mode (O(N²) operations via array resizing).
"""
import jax.numpy as jnp
from typing import Dict, Optional
from core.graph import GraphState, CapacityExceededError


def add_node(
    state: GraphState,
    node_type: int = 0,
    initial_attrs: Optional[Dict[str, float]] = None
) -> GraphState:
    """
    Add node - O(1) in capacity mode, O(N²) in dynamic mode.

    Args:
        state: Current graph state
        node_type: Type label for new node
        initial_attrs: Initial values for node attributes {attr_name: value}

    Returns:
        New GraphState with added node

    Raises:
        CapacityExceededError: When capacity mode and no inactive slots available
    """
    if not state.is_capacity_mode:
        # Dynamic mode: use append/vstack (existing behavior)
        return _add_node_dynamic(state, node_type, initial_attrs)

    # Capacity mode: activate slot (O(1))
    inactive_indices = jnp.where(state.node_types == -1)[0]

    if len(inactive_indices) == 0:
        raise CapacityExceededError(
            f"Cannot add node: capacity {state.capacity} reached"
        )

    slot_id = int(inactive_indices[0])
    new_node_types = state.node_types.at[slot_id].set(node_type)

    new_node_attrs = dict(state.node_attrs)
    for attr_name, values in state.node_attrs.items():
        init_val = initial_attrs.get(attr_name, 0.0) if initial_attrs else 0.0
        new_node_attrs[attr_name] = values.at[slot_id].set(init_val)

    return state.replace(node_types=new_node_types, node_attrs=new_node_attrs)


def _add_node_dynamic(
    state: GraphState,
    node_type: int,
    initial_attrs: Optional[Dict[str, float]]
) -> GraphState:
    """Dynamic mode implementation (backward compatible)."""
    num_nodes = state.num_nodes
    new_num_nodes = num_nodes + 1

    # Update node types
    new_node_types = jnp.append(state.node_types, node_type)

    # Update node attributes
    new_node_attrs = {}
    for attr_name, values in state.node_attrs.items():
        init_val = initial_attrs.get(attr_name, 0.0) if initial_attrs else 0.0
        new_node_attrs[attr_name] = jnp.append(values, init_val)

    # Update adjacency matrices (expand with zeros)
    new_adj_matrices = {}
    for rel_name, adj in state.adj_matrices.items():
        # Add row of zeros
        new_adj = jnp.vstack([adj, jnp.zeros((1, num_nodes))])
        # Add column of zeros
        new_adj = jnp.hstack([new_adj, jnp.zeros((new_num_nodes, 1))])
        new_adj_matrices[rel_name] = new_adj

    return state.replace(
        node_types=new_node_types,
        node_attrs=new_node_attrs,
        adj_matrices=new_adj_matrices
    )


def remove_node(state: GraphState, node_id: int) -> GraphState:
    """
    Remove node - O(1) in capacity mode, O(N²) in dynamic mode.

    Args:
        state: Current graph state
        node_id: Index of node to remove

    Returns:
        New GraphState with node removed

    Raises:
        ValueError: If node_id is invalid or node already inactive
    """
    if not state.is_capacity_mode:
        # Dynamic mode: delete from arrays
        return _remove_node_dynamic(state, node_id)

    # Capacity mode: mark inactive (O(1))
    if node_id < 0 or node_id >= state.capacity:
        raise ValueError(f"Invalid node_id: {node_id}")

    if state.node_types[node_id] == -1:
        raise ValueError(f"Node {node_id} already inactive")

    new_node_types = state.node_types.at[node_id].set(-1)

    new_node_attrs = {}
    for attr_name, values in state.node_attrs.items():
        new_node_attrs[attr_name] = values.at[node_id].set(0.0)

    new_adj_matrices = {}
    for rel_name, adj in state.adj_matrices.items():
        new_adj = adj.at[node_id, :].set(0.0)
        new_adj = new_adj.at[:, node_id].set(0.0)
        new_adj_matrices[rel_name] = new_adj

    return state.replace(
        node_types=new_node_types,
        node_attrs=new_node_attrs,
        adj_matrices=new_adj_matrices
    )


def _remove_node_dynamic(state: GraphState, node_id: int) -> GraphState:
    """Dynamic mode implementation (backward compatible)."""
    if node_id < 0 or node_id >= state.num_nodes:
        raise ValueError(f"Invalid node_id: {node_id}")

    new_node_types = jnp.delete(state.node_types, node_id)

    new_node_attrs = {}
    for attr_name, values in state.node_attrs.items():
        new_node_attrs[attr_name] = jnp.delete(values, node_id)

    new_adj_matrices = {}
    for rel_name, adj in state.adj_matrices.items():
        new_adj = jnp.delete(adj, node_id, axis=0)
        new_adj = jnp.delete(new_adj, node_id, axis=1)
        new_adj_matrices[rel_name] = new_adj

    return state.replace(
        node_types=new_node_types,
        node_attrs=new_node_attrs,
        adj_matrices=new_adj_matrices
    )


def add_edge(
    state: GraphState,
    from_id: int,
    to_id: int,
    rel_name: str = "default",
    weight: float = 1.0,
    directed: bool = False
) -> GraphState:
    """
    Add edge between nodes.

    Args:
        state: Current graph state
        from_id: Source node index
        to_id: Target node index
        rel_name: Name of adjacency matrix to update
        weight: Edge weight
        directed: If False, add edge in both directions

    Returns:
        New GraphState with edge added
    """
    if rel_name not in state.adj_matrices:
        raise ValueError(f"Unknown relation: {rel_name}")

    adj = state.adj_matrices[rel_name]
    new_adj = adj.at[from_id, to_id].set(weight)

    if not directed:
        new_adj = new_adj.at[to_id, from_id].set(weight)

    return state.update_adj_matrix(rel_name, new_adj)


def remove_edge(
    state: GraphState,
    from_id: int,
    to_id: int,
    rel_name: str = "default",
    directed: bool = False
) -> GraphState:
    """
    Remove edge between nodes.

    Args:
        state: Current graph state
        from_id: Source node index
        to_id: Target node index
        rel_name: Name of adjacency matrix to update
        directed: If False, remove edge in both directions

    Returns:
        New GraphState with edge removed
    """
    if rel_name not in state.adj_matrices:
        raise ValueError(f"Unknown relation: {rel_name}")

    adj = state.adj_matrices[rel_name]
    new_adj = adj.at[from_id, to_id].set(0.0)

    if not directed:
        new_adj = new_adj.at[to_id, from_id].set(0.0)

    return state.update_adj_matrix(rel_name, new_adj)
