# execution/functional_core/evaluator.py

from typing import Dict, Any, Optional, Callable
import jax
import jax.numpy as jnp

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState
from core.category import Transform

class TransformEvaluator:
    """
    Pure evaluation of graph transformations.
    
    The TransformEvaluator is responsible for applying transformations
    to graph states in a pure, functional manner, without side effects
    or execution optimizations.
    """
    
    def evaluate(self, transform: Transform, state: GraphState) -> GraphState:
        """
        Evaluate a transformation on a graph state.
        
        Args:
            transform: The transformation to evaluate
            state: The graph state to transform
            
        Returns:
            The resulting graph state
        """
        # For pure transformations, simply apply the function
        return transform(state)
    
    def is_parallelizable(self, transform: Transform) -> bool:
        """
        Determine if a transformation can be executed in parallel.
        
        Args:
            transform: The transformation to check
            
        Returns:
            True if the transformation can be parallelized, False otherwise
        """
        # Check for parallelizable attribute or infer from structure
        if hasattr(transform, 'parallelizable'):
            return transform.parallelizable
        
        # Default to False for safety
        return False