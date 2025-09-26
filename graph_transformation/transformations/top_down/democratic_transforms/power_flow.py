"""
Voting power calculation as a flow on the delegation graph.
"""
import jax.numpy as jnp
from typing import Dict, Any, Callable

from core.graph import GraphState

def create_power_flow_transform(
    config: Dict[str, Any] = None
) -> Callable[[GraphState], GraphState]:
    """
    Create a transformation that calculates effective voting power
    by propagating power through the delegation graph.
    
    Args:
        config: Optional configuration parameters
        
    Returns:
        A transformation function that updates voting power
    """
    config = config or {}
    
    def transform(state: GraphState) -> GraphState:
        # Skip if delegation matrix is missing
        if "delegation" not in state.adj_matrices:
            return state
            
        # Get delegation matrix and number of agents
        delegation = state.adj_matrices["delegation"]
        num_agents = delegation.shape[0]
        
        # Initialize with base voting power
        base_power = config.get("base_voting_power", 1.0)
        initial_power = jnp.ones(num_agents) * base_power
        
        # Calculate delegated power
        # First, identify who delegates (has outgoing edges)
        delegates = jnp.sum(delegation, axis=1) > 0
        
        # Those who delegate transfer their power
        direct_power = jnp.where(delegates, 0.0, initial_power)
        
        # Calculate received power through delegation
        # This is a simplification - a full implementation would handle
        # multi-level delegation by solving a system of equations
        received_power = jnp.zeros(num_agents)
        for i in range(num_agents):
            if delegates[i]:
                # Find who this agent delegates to
                for j in range(num_agents):
                    if delegation[i, j] > 0:
                        # Transfer power
                        received_power = received_power.at[j].add(initial_power[i])
        
        # Total effective voting power
        voting_power = direct_power + received_power
        
        # Update node attributes
        new_attrs = dict(state.node_attrs)
        new_attrs["voting_power"] = voting_power
        
        return state.replace(node_attrs=new_attrs)
    
    return transform