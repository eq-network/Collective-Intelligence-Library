"""
Foundational category theory concepts adapted for JAX compatibility.

This module defines the core categorical concepts like morphisms and composition,
designed to work seamlessly with JAX's functional paradigm and transformations.
"""
from typing import TypeVar, Generic, Callable, List, Set, Dict, Any, Union, TYPE_CHECKING
from functools import wraps
import jax

from .graph import GraphState

# Make Transform a type alias to avoid circular imports
Transform = Callable[[GraphState], GraphState]

# Import Property only when type checking to avoid circular imports
if TYPE_CHECKING:
    from .property import Property

def compose(f: Transform, g: Transform) -> Transform:
    """
    Compose two transformations (g ∘ f) to create a new transformation.
    
    This is the fundamental operation in category theory, allowing us to chain
    transformations while preserving their mathematical properties.
    
    Args:
        f: First transformation to apply
        g: Second transformation to apply
    
    Returns:
        A new transformation that applies f followed by g
    """
    @wraps(g)
    def composed(state: GraphState) -> GraphState:
        """Apply f followed by g."""
        return g(f(state))
    
    # Preserve metadata from the original functions
    f_props = getattr(f, 'preserves', set())
    g_props = getattr(g, 'preserves', set())
    composed.preserves = f_props.intersection(g_props)
    
    return composed

def identity() -> Transform:
    """Identity transformation: fundamental to the category of graph transformations."""
    def id_transform(state: GraphState) -> GraphState:
        return state
        
    # The identity preserves ALL properties by definition
    id_transform.preserves = "ALL_PROPERTIES"  # Special marker
    return id_transform


def sequential(*transforms: Transform) -> Transform:
    """
    Compose a sequence of transformations into a single transformation.
    
    Args:
        *transforms: A sequence of transformation functions
    
    Returns:
        A single transformation that applies all transforms in sequence
    """
    if not transforms:
        # Return identity transformation (call it to get the actual transform)
        return identity()
    
    if len(transforms) == 1:
        return transforms[0]
    
    result = transforms[0]
    for t in transforms[1:]:
        result = compose(result, t)
    
    return result


def attach_properties(transform: Transform, properties: Set[Any]) -> Transform:
    """
    Attach a set of preserved properties to a transformation function.
    
    Args:
        transform: The transformation function
        properties: Set of properties preserved by the transformation
    
    Returns:
        The same transformation function with properties attached
    """
    transform.preserves = properties
    return transform


def jit_transform(transform: Transform) -> Transform:
    """
    Apply JAX's JIT compilation to a transformation.
    
    This improves performance by compiling the transformation function.
    The compiled version preserves the same properties as the original.
    
    Args:
        transform: The transformation function to compile
    
    Returns:
        A JIT-compiled version of the transformation
    """
    jitted = jax.jit(transform)
    
    # Preserve properties
    if hasattr(transform, 'preserves'):
        jitted.preserves = transform.preserves
    
    return jitted


def validate_properties(transform: Transform, initial_state: GraphState, final_state: GraphState) -> bool:
    """
    Validate that a transformation preserves its declared properties.
    
    Args:
        transform: The transformation to validate
        initial_state: The state before transformation
        final_state: The state after transformation
    
    Returns:
        True if all preserved properties are maintained, False otherwise
    """
    if not hasattr(transform, 'preserves'):
        return True
    
    for prop in transform.preserves:
        if not prop.check(final_state):
            return False
    
    return True