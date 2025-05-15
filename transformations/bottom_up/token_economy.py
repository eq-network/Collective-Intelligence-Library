# transformations/bottom_up/token_economy.py
from typing import Dict, Any, Callable
import jax.numpy as jnp

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState

def create_token_budget_calculator(refresh_period: int = 5) -> Callable[[GraphState, Dict[str, Any]], GraphState]:
    """
    Create a calculator function that refreshes token budgets periodically.
    
    Args:
        refresh_period: Number of rounds between token budget refreshes
        
    Returns:
        A calculator function compatible with resource transform
    """
    def calculator(state: GraphState, config: Dict[str, Any]) -> GraphState:
        # Skip if token system not active
        if "token_budget" not in state.node_attrs or "tokens_spent" not in state.node_attrs:
            return state
            
        # Extract current round
        current_round = state.global_attrs.get("round", 0)
        
        # Check if we need to refresh tokens
        if current_round % refresh_period == 0 and current_round > 0:
            # Create updated node attributes
            new_node_attrs = dict(state.node_attrs)
            new_node_attrs["tokens_spent"] = jnp.zeros_like(state.node_attrs["tokens_spent"])
            return state.replace(node_attrs=new_node_attrs)
            
        return state
    
    return calculator