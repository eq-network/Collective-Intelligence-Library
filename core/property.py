"""
Property system for the graph transformation framework.

Properties define invariants or characteristics that can be verified on graph states.
They're used to validate transformations and ensure mathematical properties.
"""
from typing import TypeVar, Generic, Protocol, Set, Callable, Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
import jax.numpy as jnp

# Forward reference to avoid circular import
if TYPE_CHECKING:
    from .category import Transform

from .graph import GraphState

class Property:
    """
    A property is a predicate over graph states that can be checked.
    
    Properties form the foundation of our algebraic approach to transformations:
    - Transformations preserve sets of properties
    - Composition of transformations preserves the intersection of properties
    """
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    def check(self, state: GraphState) -> bool:
        """
        Check if the graph state satisfies this property.
        
        Returns:
            bool: True if the property holds, False otherwise
        """
        raise NotImplementedError
    
    def __and__(self, other: 'Property') -> 'Property':
        """
        Conjunction of properties (p ∧ q).
        Returns a new property that holds if both this property and the other hold.
        """
        return ConjunctiveProperty(f"{self.name} AND {other.name}", [self, other])
    
    def __or__(self, other: 'Property') -> 'Property':
        """
        Disjunction of properties (p ∨ q).
        Returns a new property that holds if either this property or the other holds.
        """
        return DisjunctiveProperty(f"{self.name} OR {other.name}", [self, other])
    
    def __invert__(self) -> 'Property':
        """
        Negation of a property (¬p).
        Returns a new property that holds if this property does not hold.
        """
        return NegatedProperty(f"NOT {self.name}", self)


class ConservesSum(Property):
    """
    Property checking that the sum of a node attribute remains constant.

    This property must be bound to an initial state before it can verify conservation.

    Example:
        # Create unbound property
        conservation = ConservesSum("resources")

        # Bind to initial state
        bound_conservation = conservation.bind(initial_state)

        # Now can check if sum is conserved
        final_state = transform(initial_state)
        assert bound_conservation.check(final_state)  # Verifies sum is same
    """

    def __init__(self, attribute_name: str, reference_sum: float = None, tolerance: float = 1e-5):
        """
        Create a ConservesSum property.

        Args:
            attribute_name: Name of the node attribute to check
            reference_sum: Reference sum to compare against (None if unbound)
            tolerance: Tolerance for floating point comparison
        """
        super().__init__(f"ConservesSum({attribute_name})")
        self.attribute_name = attribute_name
        self.reference_sum = reference_sum
        self.tolerance = tolerance

    def bind(self, initial_state: GraphState) -> 'ConservesSum':
        """
        Bind this property to an initial state's sum.

        Args:
            initial_state: State to use as reference

        Returns:
            New ConservesSum property bound to initial state's sum

        Raises:
            ValueError: If attribute doesn't exist in state
        """
        if self.attribute_name not in initial_state.node_attrs:
            raise ValueError(f"Attribute '{self.attribute_name}' not found in state")

        values = initial_state.node_attrs[self.attribute_name]

        # Use active mask in capacity mode, else sum all
        if initial_state.is_capacity_mode:
            active_mask = initial_state.get_active_mask()
            total = float(jnp.sum(values * active_mask))
        else:
            total = float(jnp.sum(values))

        return ConservesSum(self.attribute_name, reference_sum=total, tolerance=self.tolerance)

    def check(self, state: GraphState) -> bool:
        """
        Check if the total sum is conserved (within tolerance).

        Args:
            state: State to check

        Returns:
            True if sum is conserved, False otherwise

        Raises:
            ValueError: If property is not bound (call bind() first)
        """
        if self.reference_sum is None:
            raise ValueError(
                f"ConservesSum property for '{self.attribute_name}' is not bound. "
                f"Call bind(initial_state) before checking."
            )

        if self.attribute_name not in state.node_attrs:
            return False

        values = state.node_attrs[self.attribute_name]

        # Use active mask in capacity mode, else sum all
        if state.is_capacity_mode:
            active_mask = state.get_active_mask()
            current_sum = float(jnp.sum(values * active_mask))
        else:
            current_sum = float(jnp.sum(values))

        return bool(jnp.isclose(current_sum, self.reference_sum, rtol=self.tolerance))


class ConjunctiveProperty(Property):
    """A property that is the conjunction of multiple properties."""
    
    def __init__(self, name: str, properties: list[Property]):
        super().__init__(name)
        self.properties = properties
    
    def check(self, state: GraphState) -> bool:
        """A conjunctive property holds if all constituent properties hold."""
        return all(prop.check(state) for prop in self.properties)


class DisjunctiveProperty(Property):
    """A property that is the disjunction of multiple properties."""
    
    def __init__(self, name: str, properties: list[Property]):
        super().__init__(name)
        self.properties = properties
    
    def check(self, state: GraphState) -> bool:
        """A disjunctive property holds if any constituent property holds."""
        return any(prop.check(state) for prop in self.properties)


class NegatedProperty(Property):
    """A property that is the negation of another property."""
    
    def __init__(self, name: str, property: Property):
        super().__init__(name)
        self.property = property
    
    def check(self, state: GraphState) -> bool:
        """A negated property holds if the original property does not hold."""
        return not self.property.check(state)
    
class PropertyCategory:
    """Represents a subcategory of transformations preserving specific properties."""
    
    def __init__(self, name: str, properties: Set[Property]):
        self.name = name
        self.properties = properties
    
    def contains(self, transform: 'Transform') -> bool:
        """Check if a transformation belongs to this category."""
        if not hasattr(transform, 'preserves'):
            return False
        return all(p in transform.preserves for p in self.properties)
    
    def __call__(self, transform: 'Transform') -> 'Transform':
        """Decorator that marks a transformation as belonging to this category."""
        if not hasattr(transform, 'preserves'):
            transform.preserves = set()
        transform.preserves = transform.preserves.union(self.properties)
        return transform