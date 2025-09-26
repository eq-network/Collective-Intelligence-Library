# transformations/bottom_up/portfolio_analysis.py
import jax.numpy as jnp
from typing import Callable, Dict, Any

from core.graph import GraphState

# Pure transformation generator - domain-agnostic, just handles GraphState
def create_portfolio_analyzer(
    analysis_function: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
) -> Callable[[GraphState], GraphState]:
    """
    Creates a transformation that analyzes portfolios using an injected function.
    
    Args:
        analysis_function: Function that produces agent preferences given:
            1. Agent attributes
            2. Global portfolio/market information
    
    Returns:
        A transformation from GraphState to GraphState
    """
    def transform(state: GraphState) -> GraphState:
        # Extract state without domain assumptions
        if "token_budget" not in state.node_attrs or "portfolios" not in state.global_attrs:
            return state
            
        # Apply the injected analysis function to each agent
        num_agents = state.num_nodes
        portfolio_names = list(state.global_attrs["portfolios"].keys())
        portfolio_prefs = jnp.zeros((num_agents, len(portfolio_names)))
        token_spent = jnp.zeros(num_agents)
        
        # Process each agent
        for i in range(num_agents):
            # Create agent context for analysis function
            agent_attrs = {k: v[i] for k, v in state.node_attrs.items()}
            global_info = {
                "portfolios": state.global_attrs["portfolios"],
                "prediction_market": state.global_attrs.get("prediction_market"),
                "round": state.global_attrs.get("round", 0)
            }
            
            # Call the injected analysis function
            result = analysis_function(agent_attrs, global_info)
            
            # Update preferences and tokens
            if "preferences" in result:
                for j, name in enumerate(portfolio_names):
                    if name in result["preferences"]:
                        portfolio_prefs = portfolio_prefs.at[i, j].set(result["preferences"][name])
            
            if "tokens_spent" in result:
                token_spent = token_spent.at[i].set(result["tokens_spent"])
        
        # Update state
        new_node_attrs = dict(state.node_attrs)
        new_node_attrs["portfolio_preferences"] = portfolio_prefs
        
        # Only update tokens_spent if it already exists
        if "tokens_spent" in state.node_attrs:
            current_spent = state.node_attrs["tokens_spent"]
            new_node_attrs["tokens_spent"] = current_spent + token_spent
        
        return state.replace(node_attrs=new_node_attrs)
    
    return transform