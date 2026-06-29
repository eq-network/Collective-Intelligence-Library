"""
Core module for the Collective Intelligence Library graph transformation framework.

Fundamental building blocks:
- GraphState: Immutable JAX pytree graph representation
- Transform: Pure functions GraphState -> GraphState
- Category theory operations: compose, sequential, identity, parallel, conditional
- Environment: lax.scan-based simulation runner
- Pure compiled scan tier: run_scan / run_scan_batch
"""

from .graph import GraphState
from .category import (
    Transform, transform, compose, sequential, identity, parallel, conditional,
    jit_transform, attach_properties,
    gated, bind_time, TimeAware,
)
from .pipeline import compile_pipeline, validate_pipeline, get_execution_order
from .environment import Environment
from .scan import run_scan, run_scan_batch, RoundFn, TraceFn
from .property import Property, ConservesSum
from .initialization import initialize_graph_state
from .agents import Policy, get_observation, apply_action
from .protocols import PureAgent, Mechanism

__all__ = [
    'GraphState',
    'Transform',
    'transform',
    'compile_pipeline',
    'validate_pipeline',
    'get_execution_order',
    'compose',
    'sequential',
    'identity',
    'parallel',
    'conditional',
    'jit_transform',
    'attach_properties',
    'gated',
    'bind_time',
    'TimeAware',
    'Environment',

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
    'PureAgent',
    'Mechanism',
    'get_observation',
    'apply_action',
]
