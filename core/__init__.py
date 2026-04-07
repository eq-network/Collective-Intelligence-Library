"""
Core module for the Mycorrhiza graph transformation framework.

Fundamental building blocks:
- GraphState: Immutable JAX pytree graph representation
- Transform: Pure functions GraphState -> GraphState
- Category theory operations: compose, sequential, identity, parallel, conditional
- Environment: lax.scan-based simulation runner
"""

from .graph import GraphState
from .category import (
    Transform, compose, sequential, identity, parallel, conditional,
    jit_transform, attach_properties,
)
from .environment import Environment

__all__ = [
    'GraphState',
    'Transform',
    'compose',
    'sequential',
    'identity',
    'parallel',
    'conditional',
    'jit_transform',
    'attach_properties',
    'Environment',
]
