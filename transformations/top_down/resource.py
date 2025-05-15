"""
Resource application transformation for applying collective decisions.
"""
import jax.numpy as jnp
from typing import Dict, Any, Callable

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState

def create_resource_transform(
    resource_calculator: Callable[[GraphState, Dict[str, Any]], float] = None,
    config: Dict[str, Any] = None
) -> Callable[[GraphState], GraphState]:
    """
    Create a transformation that applies the results of decisions to resources.
    
    Args:
        resource_calculator: Function that calculates resource changes based on decisions.
                            If None, uses a simple default calculation.
        config: Optional configuration parameters
        
    Returns:
        A transformation function that updates resources based on decisions
    """
    config = config or {}
    
    # Simple default resource calculator
    def default_calculator(state: GraphState, config: Dict[str, Any]) -> float:
        # Default to small positive change if no specific logic provided
        decision = state.global_attrs.get("current_decision", 0)
        options = config.get("num_options", 1)
        
        # Simple strategy: fixed payoff based on decision
        return 1.0 + (decision / options) * 0.2  # 0.2 = max 20% growth
    
    # Use provided calculator or default
    calculator = resource_calculator or default_calculator
    
    def transform(state: GraphState) -> GraphState:
        # Skip if no decision has been made
        if "current_decision" not in state.global_attrs:
            return state
            
        # Get current resources
        current_resources = state.global_attrs.get("total_resources", 100.0)
        
        # Calculate resource change factor
        change_factor = calculator(state, config)
        
        # Update resources
        new_resources = current_resources * change_factor
        
        # Update global attributes
        new_globals = dict(state.global_attrs)
        new_globals["total_resources"] = float(new_resources)
        new_globals["last_resource_change"] = float(change_factor)
        
        # Track history if configured
        if config.get("track_history", True):
            history = new_globals.get("resource_history", [])
            history.append({
                "round": new_globals.get("round", 0),
                "decision": int(new_globals["current_decision"]),
                "change_factor": float(change_factor),
                "resources": float(new_resources)
            })
            new_globals["resource_history"] = history
        
        return state.replace(global_attrs=new_globals)
    
    return transform