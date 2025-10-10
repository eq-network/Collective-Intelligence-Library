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
    """Resource transform with CLEAN debug output."""
    config = config or {}
    calculator = resource_calculator or default_calculator
    resource_attr_name = config.get("resource_attr_name", "current_total_resources")
    history_attr_name = config.get("history_attr_name", "resource_history")
    
    def transform(state: GraphState) -> GraphState:
        if "current_decision" not in state.global_attrs:
            return state
            
        current_resources = state.global_attrs.get(resource_attr_name, 100.0)
        change_factor = calculator(state, config)
        new_resources = current_resources * change_factor
        
        # CLEAN DEBUG: Concise resource update information
        round_num = state.global_attrs.get("round_num", 0)
        decision_idx = state.global_attrs.get("current_decision", -1)
        print(f"[R{round_num:02d}] Resources: {current_resources:.1f} → {new_resources:.1f} "
              f"(×{change_factor:.3f}, Decision: {decision_idx})")
        
        # Update global attributes
        new_globals = dict(state.global_attrs)
        new_globals[resource_attr_name] = float(new_resources)
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