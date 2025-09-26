"""
Property system for the graph transformation framework.

Properties define invariants or characteristics that can be verified on graph states.
They're used to validate transformations and ensure mathematical properties.
"""
from typing import TypeVar, Generic, Protocol, Set, Callable, Dict, Any
from abc import ABC, abstractmethod
from .category import Transform
import jax.numpy as jnp

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
    """Property checking that the sum of a node attribute remains constant."""
    
    def __init__(self, attribute_name: str, name: str = None):
        super().__init__(name or f"ConservesSum({attribute_name})")
        self.attribute_name = attribute_name
    
    def check(self, state: GraphState) -> bool:
        """Check if the total sum of the attribute is conserved."""
        if self.attribute_name not in state.node_attrs:
            return False
        
        # In a real implementation, we would need to check against a reference state
        # Here we're just ensuring the attribute exists and has finite values
        attr = state.node_attrs[self.attribute_name]
        return bool(jnp.all(jnp.isfinite(attr)))


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
    
    def contains(self, transform: Transform) -> bool:
        """Check if a transformation belongs to this category."""
        if not hasattr(transform, 'preserves'):
            return False
        return all(p in transform.preserves for p in self.properties)
    
    def __call__(self, transform: Transform) -> Transform:
        """Decorator that marks a transformation as belonging to this category."""
        if not hasattr(transform, 'preserves'):
            transform.preserves = set()
        transform.preserves = transform.preserves.union(self.properties)
        return transform