"""
Voting transformations that aggregate agent preferences into collective decisions.
"""
import jax.numpy as jnp
from typing import Dict, Any, Callable

from core.graph import GraphState

def create_voting_transform(
    vote_aggregator: Callable[[GraphState, Dict[str, Any]], jnp.ndarray] = None,
    config: Dict[str, Any] = None
) -> Callable[[GraphState], GraphState]:
    """
    Create a transformation that aggregates individual votes into a collective decision.
    
    Args:
        vote_aggregator: Function that determines how votes are aggregated.
                         If None, uses simple averaging of beliefs.
        config: Optional configuration parameters
        
    Returns:
        A transformation function that updates state with voting results
    """
    config = config or {}
    
    # Default aggregator uses simple averaging of beliefs
    def default_aggregator(state: GraphState, config: Dict[str, Any]) -> jnp.ndarray:
        if "belief" not in state.node_attrs:
            return jnp.ones(1)  # Fallback
            
        # Simple average of all beliefs
        return jnp.mean(state.node_attrs["belief"], axis=0)
    
    # Use provided aggregator or default
    aggregator = vote_aggregator or default_aggregator
    
    def transform(state: GraphState) -> GraphState:
        # Apply the aggregation strategy
        vote_distribution = aggregator(state, config)
        
        # Determine the winning option
        decision = jnp.argmax(vote_distribution)
        
        # Update global attributes with decision
        new_globals = dict(state.global_attrs)
        new_globals["current_decision"] = int(decision)
        new_globals["vote_distribution"] = vote_distribution
        
        return state.replace(global_attrs=new_globals)
    
    return transform