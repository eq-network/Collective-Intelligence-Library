"""
Foundational category theory concepts adapted for JAX compatibility.

This module defines the core categorical concepts like morphisms and composition,
designed to work seamlessly with JAX's functional paradigm and transformations.
"""
from typing import TypeVar, Generic, Callable, List, Set, Dict, Any, Union, TYPE_CHECKING, FrozenSet, Optional
from functools import wraps
import jax

from .graph import GraphState

# Make Transform a type alias to avoid circular imports
Transform = Callable[[GraphState], GraphState]


def transform(
    reads: List[str],
    writes: List[str],
    name: Optional[str] = None,
) -> Callable[[Transform], Transform]:
    """
    Decorator that attaches read/write set metadata to a transform.

    The read/write sets declare which GraphState fields this transform
    accesses. The pipeline compiler uses these to derive execution order
    (topological sort) and detect parallelizable groups (disjoint writes).

    Usage:
        @transform(reads=["vote_weights"], writes=["harvest_target", "penalty_lambda"])
        def direct_democracy(state: GraphState) -> GraphState:
            ...

    The decorated function gains:
        .reads:  frozenset of field names read
        .writes: frozenset of field names written
        .name:   human-readable name (defaults to function name)
    """
    reads_set = frozenset(reads)
    writes_set = frozenset(writes)

    def decorator(fn: Transform) -> Transform:
        @wraps(fn)
        def wrapper(state: GraphState) -> GraphState:
            return fn(state)

        wrapper.reads = reads_set
        wrapper.writes = writes_set
        wrapper.name = name or fn.__name__
        # Preserve any existing property metadata
        if hasattr(fn, 'preserves'):
            wrapper.preserves = fn.preserves
        return wrapper

    return decorator

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
    # Handle "ALL_PROPERTIES" sentinel: identity preserves everything
    if f_props == "ALL_PROPERTIES":
        composed.preserves = g_props
    elif g_props == "ALL_PROPERTIES":
        composed.preserves = f_props
    else:
        composed.preserves = f_props.intersection(g_props)

    # Propagate read/write metadata through composition
    composed.reads = getattr(f, 'reads', frozenset()) | getattr(g, 'reads', frozenset())
    composed.writes = getattr(f, 'writes', frozenset()) | getattr(g, 'writes', frozenset())
    composed.name = f"{getattr(f, 'name', '?')} >> {getattr(g, 'name', '?')}"

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


def parallel(*transforms: Transform) -> Transform:
    """
    Execute transforms independently on the SAME input state, merge results.

    Each transform receives the same input state. The output is constructed by:
    - Taking written fields from each transform's output
    - Taking unwritten fields from the input state

    Requires all transforms to have .writes metadata (from @transform decorator).
    Raises ValueError if write sets overlap.
    """
    transform_list = list(transforms)

    for i, t in enumerate(transform_list):
        if not hasattr(t, 'writes'):
            raise ValueError(
                f"Transform at index {i} ({getattr(t, 'name', '?')}) "
                f"lacks .writes metadata — use @transform decorator"
            )

    # Check pairwise disjoint write sets
    for i in range(len(transform_list)):
        for j in range(i + 1, len(transform_list)):
            overlap = transform_list[i].writes & transform_list[j].writes
            if overlap:
                raise ValueError(
                    f"Write sets overlap between '{getattr(transform_list[i], 'name', i)}' "
                    f"and '{getattr(transform_list[j], 'name', j)}': {overlap}"
                )

    all_reads = frozenset().union(*(t.reads for t in transform_list))
    all_writes = frozenset().union(*(t.writes for t in transform_list))
    names = [getattr(t, 'name', '?') for t in transform_list]

    def merged(state: GraphState) -> GraphState:
        results = [t(state) for t in transform_list]

        new_node_attrs = dict(state.node_attrs)
        new_edge_attrs = dict(state.edge_attrs)
        new_adj_matrices = dict(state.adj_matrices)
        new_global_attrs = dict(state.global_attrs)

        for t, result in zip(transform_list, results):
            for key in t.writes:
                if key in result.node_attrs:
                    new_node_attrs[key] = result.node_attrs[key]
                if key in result.edge_attrs:
                    new_edge_attrs[key] = result.edge_attrs[key]
                if key in result.adj_matrices:
                    new_adj_matrices[key] = result.adj_matrices[key]
                if key in result.global_attrs:
                    new_global_attrs[key] = result.global_attrs[key]

        return state.replace(
            node_attrs=new_node_attrs,
            edge_attrs=new_edge_attrs,
            adj_matrices=new_adj_matrices,
            global_attrs=new_global_attrs,
        )

    merged.reads = all_reads
    merged.writes = all_writes
    merged.name = f"parallel({', '.join(names)})"
    return merged


def conditional(
    predicate: Callable[[GraphState], Any],
    transform_fn: Transform,
    predicate_reads: Optional[List[str]] = None,
) -> Transform:
    """
    Gate a transform on a predicate. If predicate(state) is true, apply transform.
    Otherwise, return state unchanged.

    Uses jax.lax.cond for JIT compatibility — both branches are traced,
    but only one executes.

    Args:
        predicate: Function returning a JAX scalar boolean.
        transform_fn: Transform to apply when predicate is true.
        predicate_reads: Fields the predicate reads (for pipeline ordering).
    """
    def gated(state: GraphState) -> GraphState:
        return jax.lax.cond(predicate(state), transform_fn, lambda s: s, state)

    pred_reads = frozenset(predicate_reads) if predicate_reads else frozenset()
    gated.reads = getattr(transform_fn, 'reads', frozenset()) | pred_reads
    gated.writes = getattr(transform_fn, 'writes', frozenset())
    gated.name = f"conditional:{getattr(transform_fn, 'name', '?')}"
    return gated