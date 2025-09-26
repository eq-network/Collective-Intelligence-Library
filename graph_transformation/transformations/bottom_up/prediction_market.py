# transformations/bottom_up/prediction_market.py
"""
Prediction market transformation - a general mechanism for incorporating
prediction signals into the graph state.

This transform follows the inversion of control principle, accepting a prediction
generator function rather than embedding domain-specific prediction logic.
"""
import jax.numpy as jnp
from typing import Dict, Any, Callable

from core.graph import GraphState

def create_prediction_market_transform(
    prediction_generator: Callable[[GraphState, Dict[str, Any]], jnp.ndarray] = None,
    config: Dict[str, Any] = None
) -> Callable[[GraphState], GraphState]:
    """
    Create a transformation that incorporates prediction market signals into the graph state.
    
    Args:
        prediction_generator: Function that generates predictions based on the current state.
                             If None, a default generator using global attributes will be used.
        config: Additional configuration parameters
        
    Returns:
        A transformation that adds prediction market signals to the graph state
    """
    config = config or {}
    
    # Default prediction generator that uses global attributes
    def default_generator(state: GraphState, config: Dict[str, Any]) -> jnp.ndarray:
        """Default implementation that expects ground truth in global attributes."""
        # This is just a simple placeholder - actual implementations would be provided
        # by the system integrator based on specific experimental needs
        return jnp.ones(config.get("num_options", 1))
    
    # Use provided generator or default
    generator = prediction_generator or default_generator
    
    def transform(state: GraphState) -> GraphState:
        # Generate predictions using the provided generator
        predictions = generator(state, config)
        
        # Ensure predictions is a proper JAX array
        if not isinstance(predictions, jnp.ndarray):
            predictions = jnp.array(predictions)
        
        # Update global attributes with prediction market results
        new_global_attrs = dict(state.global_attrs)
        new_global_attrs["prediction_market"] = predictions
        
        return state.replace(global_attrs=new_global_attrs)
    
    return transform