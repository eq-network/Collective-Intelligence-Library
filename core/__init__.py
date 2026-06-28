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
<<<<<<< Updated upstream
    Transform, compose, sequential, identity, parallel, conditional,
    jit_transform, attach_properties,
)
from .environment import Environment
=======
    Transform, compose, sequential, identity, jit_transform, attach_properties,
    gated, bind_time, TimeAware,
)
from .scan import run_scan, run_scan_batch, RoundFn, TraceFn
from .property import Property, ConservesSum
from .initialization import initialize_graph_state
from .agents import Policy, get_observation, apply_action
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    'Environment',
=======
    'gated',
    'bind_time',
    'TimeAware',

    # Pure compiled scan tier
    'run_scan',
    'run_scan_batch',
    'RoundFn',
    'TraceFn',

    # Properties
    'Property',
    'ConservesSum',

    # Initialization
    'initialize_graph_state',

    # Agents
    'Policy',
    'get_observation',
    'apply_action',
>>>>>>> Stashed changes
]
