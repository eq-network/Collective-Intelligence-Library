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

# In transformations/top_down/resource.py
def create_resource_transform(
    resource_calculator: Callable[[GraphState, Dict[str, Any]], float] = None,
    config: Dict[str, Any] = None
) -> Callable[[GraphState], GraphState]:
    config = config or {}
    
    # Use provided calculator or default
    calculator = resource_calculator or default_calculator
    
    # Extract configuration parameter with sensible default - critical fix!
    resource_attr_name = config.get("resource_attr_name", "total_resources")
    history_attr_name = config.get("history_attr_name", "resource_history")
    
    def transform(state: GraphState) -> GraphState:
        # Skip if no decision has been made
        if "current_decision" not in state.global_attrs:
            return state
            
        # Use configurable attribute name
        current_resources = state.global_attrs.get(resource_attr_name, 100.0)
        
        # Calculate resource change factor
        change_factor = calculator(state, config)
        
        # Update resources
        new_resources = current_resources * change_factor
        
        # Add debug output to track resource updates
        print(f"DEBUG: Resource update [{resource_attr_name}]: {current_resources} * {change_factor} = {new_resources}")
        
        # Update global attributes with configurable name
        new_globals = dict(state.global_attrs)
        new_globals[resource_attr_name] = float(new_resources)
        
        # CRITICAL: Also update "total_resources" attribute for metric collectors
        new_globals["total_resources"] = float(new_resources)
        
        new_globals["last_resource_change"] = float(change_factor)
        
        # Track history if configured
        if config.get("track_history", True):
            history = new_globals.get(history_attr_name, [])
            history.append({
                "round": new_globals.get("round", 0),
                "decision": int(new_globals["current_decision"]),
                "change_factor": float(change_factor),
                "resources": float(new_resources)
            })
            new_globals[history_attr_name] = history
        
        return state.replace(global_attrs=new_globals)
    
    return transform