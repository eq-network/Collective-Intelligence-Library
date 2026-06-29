"""Environments catalog — reusable, governance-agnostic simulation substrates.

Type function:  ``EnvFactory = (**cfg) -> EnvSpec`` (see ``.spec``). The catalog is the
explicit ``REGISTRY`` dict below — open this file to see every environment. ``make_env``
and ``list_envs`` are thin conveniences over it.

    from cilib.environments import make_env, list_envs
    env = make_env("commons_harvest", n_agents=16, grid=(18, 18))
    finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=64, n_steps=1000)
    scores = env.evaluate(traces)        # GovSim suite
"""
from __future__ import annotations

from .spec import EnvSpec, MetricFn
from . import commons_metrics            # noqa: F401  (shared GovSim metric helpers)
from .commons_harvest import build_commons_harvest

# name -> builder ((**cfg) -> EnvSpec).
REGISTRY = {
    "commons_harvest": build_commons_harvest,
}


def make_env(name: str, **cfg) -> EnvSpec:
    """Instantiate a registered environment by name, passing ``cfg`` to its builder."""
    if name not in REGISTRY:
        raise KeyError(f"unknown environment {name!r}; registered: {list_envs()}")
    return REGISTRY[name](**cfg)


def list_envs():
    """Sorted names of all registered environments."""
    return sorted(REGISTRY)


__all__ = ["EnvSpec", "MetricFn", "commons_metrics", "REGISTRY", "make_env", "list_envs"]
