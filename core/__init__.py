"""
Core module for the Mycorrhiza graph transformation framework.

This module provides the fundamental building blocks:
- GraphState: Immutable graph representation
- Transform: Pure functions on GraphState
- Category theory operations: compose, sequential, identity
- Property system: Invariants and verification
- Initialization: Creating initial graph states
"""

from .graph import GraphState
from .category import Transform, compose, sequential, identity, jit_transform, attach_properties
from .property import Property, ConservesSum
from .initialization import initialize_graph_state
from .agents import Policy, get_observation, apply_action

__all__ = [
    # Graph state
    'GraphState',

    # Category theory
    'Transform',
    'compose',
    'sequential',
    'identity',
    'jit_transform',
    'attach_properties',

    # Properties
    'Property',
    'ConservesSum',

    # Initialization
    'initialize_graph_state',

    # Agents
    'Policy',
    'get_observation',
    'apply_action',
]
