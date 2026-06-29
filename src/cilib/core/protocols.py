"""Type functions for the catalogs — the contracts every building block satisfies.

A *catalog* (``agents`` / ``transformations`` / ``mechanisms`` / ``environments``)
is a registry of factories that share one shape: ``Config -> TypedCallable``, where
the callable satisfies one of the protocols below. That shared shape is what makes
any entry swappable for a sibling.

These are **structural** (``Protocol`` / type aliases), not base classes: a block
conforms by shape, not by inheritance. This module is the single source of truth —
every catalog imports its contract from here.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

# Canonical callables, re-exported so there is one place to look.
from .category import Transform, TimeAware  # Transform: state->state; TimeAware: (state,t,key)->state
from .agents import Policy                  # Policy: (obs, key) -> action


@runtime_checkable
class PureAgent(Protocol):
    """A side-effect-free agent: yields the pure round it implements, runnable
    inside ``core.scan.run_scan`` and ``vmap``-able over seeds."""
    def round_fn(self) -> TimeAware: ...


# A mechanism is a typed Transform — a Transform that declares its ``.reads`` /
# ``.writes`` via the ``@transform`` decorator so ``compile_pipeline`` can order it.
Mechanism = Transform

# The "type function" of each catalog: Config -> TypedCallable.
TransformFactory = Callable[..., Transform]   # transformations/ and mechanisms/
AgentFactory = Callable[..., Policy]           # agents/  (a PureAgent factory also qualifies)
EnvFactory = Callable[..., Any]                # environments/  (**cfg -> EnvSpec)

__all__ = [
    "Transform", "TimeAware", "Policy", "PureAgent", "Mechanism",
    "TransformFactory", "AgentFactory", "EnvFactory",
]
