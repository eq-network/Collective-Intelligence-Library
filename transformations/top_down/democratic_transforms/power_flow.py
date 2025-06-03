"""
Voting power calculation as a flow on the delegation graph.
Handles transitive delegation iteratively.
"""
import jax.numpy as jnp
from typing import Dict, Any, Callable

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent.parent # Adjusted for this file's location
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState

def create_power_flow_transform(
    config: Dict[str, Any] = None,
    max_iterations: int = 100, # Max iterations for convergence
    tolerance: float = 1e-6    # Tolerance for convergence
) -> Callable[[GraphState], GraphState]:
    """
    Create a transformation that calculates effective voting power
    by propagating power transitively through the delegation graph.

    Args:
        config: Optional configuration parameters.
                Can include "base_voting_power" (float, default 1.0).
        max_iterations: Maximum number of iterations for the power flow calculation.
        tolerance: Convergence tolerance for the iterative calculation.

    Returns:
        A transformation function that updates voting power.
    """
    config = config or {}

    def transform(state: GraphState) -> GraphState:
        # Skip if delegation matrix is missing
        # Check if 'delegation' is the correct key for your adj_matrices for delegation
        # Based on test_delegation.py, it should be "delegation".
        # Based on mechanism_factory.py, it could be "delegation_graph" on initialization.
        # Ensure consistency or check for both. For now, assuming "delegation".
        if "delegation" not in state.adj_matrices:
            # If not present, maybe initialize all with base power?
            # Or return state unchanged if delegation is fundamental to this step.
            # For now, let's assume if no "delegation" matrix, everyone has base power.
            num_agents_fallback = state.num_nodes
            if num_agents_fallback > 0:
                base_power_val = config.get("base_voting_power", 1.0)
                voting_power_fallback = jnp.ones(num_agents_fallback) * base_power_val
                new_attrs_fallback = dict(state.node_attrs)
                new_attrs_fallback["voting_power"] = voting_power_fallback
                return state.replace(node_attrs=new_attrs_fallback)
            return state # No nodes, or no delegation matrix and no nodes

        delegation_matrix = state.adj_matrices["delegation"] # Shape: (num_agents, num_agents)
                                                           # delegation_matrix[i, j] = 1 if i delegates to j
        num_agents = delegation_matrix.shape[0]

        if num_agents == 0:
            new_attrs = dict(state.node_attrs)
            new_attrs["voting_power"] = jnp.array([], dtype=jnp.float32)
            return state.replace(node_attrs=new_attrs)

        base_power = config.get("base_voting_power", 1.0)
        
        # Initialize voting_power: everyone starts with their base_power.
        # This power will be transferred if they delegate.
        current_power = jnp.ones(num_agents) * base_power

        # Identify agents who are delegating (have an outgoing delegation)
        # An agent i delegates if row i in delegation_matrix has any 1s.
        is_delegating_own_power = jnp.sum(delegation_matrix, axis=1) > 0

        # Iteratively distribute power
        for iteration in range(max_iterations):
            previous_power = current_power
            power_to_distribute = current_power
            
            # Agents who are delegating will pass on ALL their current power.
            # Their "retained" power for this iteration step becomes 0 before receiving new power.
            # Agents not delegating keep their power_to_distribute.
            retained_power_step = jnp.where(is_delegating_own_power, 0.0, power_to_distribute)

            # Calculate power received from others
            # delegation_matrix.T[j, i] = 1 if i delegates to j (i.e., j receives from i)
            # So, received_power_from_others[j] = sum(power_to_distribute[i] for all i that delegate to j)
            # This is a matrix multiplication: delegation_matrix_transposed @ power_to_distribute_by_delegators
            # where power_to_distribute_by_delegators is only non-zero for those who are delegating.
            
            # Power that actual delegators are passing on:
            power_being_passed_on = jnp.where(is_delegating_own_power, power_to_distribute, 0.0)
            
            received_power_from_others = delegation_matrix.T @ power_being_passed_on
            
            current_power = retained_power_step + received_power_from_others

            # Check for convergence
            if jnp.allclose(current_power, previous_power, atol=tolerance):
                # print(f"Power flow converged in {iteration + 1} iterations.") # Optional: for debugging
                break
        # else: # Optional: for debugging if it doesn't converge
            # print(f"Power flow did not converge after {max_iterations} iterations.")

        # Update node attributes
        new_attrs = dict(state.node_attrs)
        new_attrs["voting_power"] = current_power

        return state.replace(node_attrs=new_attrs)

    return transform