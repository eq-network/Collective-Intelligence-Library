# execution/simulation.py

"""
Simulation execution module for the graph transformation framework.

This module provides functions for executing simulations by applying
transformations to graph states over multiple rounds.
"""
from typing import Dict, List, Tuple, Callable, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jr

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState
from core.category import Transform
from execution.call import execute, execute_with_instrumentation

# Type alias for PRNG keys
RandomKey = jnp.ndarray


def run_simulation(
    initial_state: GraphState,
    transform: Transform,
    num_rounds: int,
    key: RandomKey,
    execution_spec: Optional[Dict[str, Any]] = None,
    termination_condition: Optional[Callable[[GraphState], bool]] = None
) -> Tuple[GraphState, List[GraphState]]:
    """
    Execute a simulation by repeatedly applying a transformation.
    
    Args:
        initial_state: Starting graph state
        transform: Transformation to apply each round
        num_rounds: Maximum number of rounds to simulate
        key: JAX PRNG key for randomization
        execution_spec: Specifications for transformation execution
        termination_condition: Optional function that returns True
                               when simulation should stop
                              
    Returns:
        Tuple of (final_state, state_history)
    """
    # Default termination condition
    if termination_condition is None:
        def termination_condition(state: GraphState) -> bool:
            return state.global_attrs.get("total_resources", 0) < \
                   state.global_attrs.get("resource_min_threshold", 0)
    
    # Default execution specification
    if execution_spec is None:
        execution_spec = {
            "strategy": "sequential",
            "verify_properties": True,
            "track_history": True,
            "collect_metrics": True
        }
    
    # Initialize state history with initial state
    state_history = [initial_state]
    current_state = initial_state
    
    # Split random key for each round
    subkeys = jr.split(key, num_rounds + 1)
    
    # Execute rounds
    for round_num in range(1, num_rounds + 1):
        # Update round counter in state
        current_state = current_state.update_global_attr("round", round_num)
        
        # Apply transformation using execution call
        current_state = execute(transform, current_state, execution_spec)
        
        # Append to history
        state_history.append(current_state)
        
        # Check termination condition
        if termination_condition(current_state):
            break
    
    return current_state, state_history