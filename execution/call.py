# execution/call.py

from typing import Dict, List, Any, Optional, Callable, TypeVar, Protocol
import jax
import jax.numpy as jnp

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from algebra.graph import GraphState
from algebra.category import Transform
from execution.engine import ExecutionEngine

# Type for execution specifications
ExecutionSpec = Dict[str, Any]

def execute(
    transform: Transform, 
    state: GraphState, 
    spec: Optional[ExecutionSpec] = None
) -> GraphState:
    """
    Execute a transformation with specific execution parameters.
    
    This is the primary interface between environment layer and execution layer,
    providing a clean boundary for execution calls.
    
    Args:
        transform: The transformation to execute
        state: Current graph state
        spec: Execution specifications (strategy, instrumentation, etc.)
        
    Returns:
        Transformed graph state
    """
    # Default execution specification
    default_spec = {
        "strategy": "sequential",
        "verify_properties": True,
        "track_history": False,
        "collect_metrics": False
    }
    
    # Merge with provided spec
    effective_spec = {**default_spec, **(spec or {})}
    
    # Create execution engine with appropriate configuration
    engine_config = {
        "use_parallel": effective_spec["strategy"] == "parallel",
        "use_hardware_acceleration": effective_spec["strategy"] == "hardware",
        "use_memory_management": effective_spec.get("manage_memory", False),
        "track_history": effective_spec["track_history"],
        "collect_metrics": effective_spec["collect_metrics"],
    }
    
    # Configure additional parameters based on strategy
    if effective_spec["strategy"] == "parallel":
        engine_config["num_workers"] = effective_spec.get("num_workers", 4)
    elif effective_spec["strategy"] == "hardware":
        engine_config["hardware_device"] = effective_spec.get("device", "cpu")
    
    # Create engine and execute transformation
    engine = ExecutionEngine(engine_config)
    return engine.apply(
        transform, 
        state, 
        verify_properties=effective_spec["verify_properties"]
    )

def execute_with_instrumentation(
    transform: Transform, 
    state: GraphState, 
    spec: Optional[ExecutionSpec] = None
) -> tuple[GraphState, Dict[str, Any]]:
    """
    Execute a transformation and return both result and instrumentation data.
    
    Args:
        transform: The transformation to execute
        state: Current graph state
        spec: Execution specifications
        
    Returns:
        Tuple of (transformed state, instrumentation data)
    """
    # Ensure instrumentation is enabled
    effective_spec = spec or {}
    effective_spec["track_history"] = True
    effective_spec["collect_metrics"] = True
    
    # Create execution engine
    engine_config = {
        "use_parallel": effective_spec.get("strategy") == "parallel",
        "use_hardware_acceleration": effective_spec.get("strategy") == "hardware",
        "use_memory_management": effective_spec.get("manage_memory", False),
        "track_history": True,
        "collect_metrics": True,
    }
    
    engine = ExecutionEngine(engine_config)
    
    # Execute transformation
    result = engine.apply(transform, state)
    
    # Collect instrumentation data
    instrumentation = {
        "metrics": engine.get_metrics(),
        "history": engine.get_history()
    }
    
    return result, instrumentation

def batch_execute(
    transform: Transform,
    states: List[GraphState],
    spec: Optional[ExecutionSpec] = None
) -> List[GraphState]:
    """
    Execute a transformation on multiple states in batch.
    
    Args:
        transform: The transformation to execute
        states: List of graph states
        spec: Execution specifications
        
    Returns:
        List of transformed states
    """
    # Effective specification
    effective_spec = spec or {}
    
    # Create execution engine
    engine_config = {
        "use_parallel": effective_spec.get("strategy", "parallel") == "parallel",
        "num_workers": effective_spec.get("num_workers", 4),
        "track_history": False,  # History tracking not supported in batch mode
        "collect_metrics": effective_spec.get("collect_metrics", False)
    }
    
    engine = ExecutionEngine(engine_config)
    
    # Execute transformations
    results = []
    for state in states:
        results.append(engine.apply(transform, state))
    
    return results