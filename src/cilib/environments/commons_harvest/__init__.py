"""
Commons Harvest — the spatial tragedy-of-the-commons substrate (Melting Pot / SocialJax idiom).

This package exposes the ``"commons_harvest"`` builder, registered in the catalog via
``cilib.environments.REGISTRY`` and reachable through ``cilib.environments.make_env(...)``.
"""
from __future__ import annotations

from ..spec import EnvSpec
from .config import HarvestConfig, layout
from .state import make_state, make_init_fn
from .dynamics import make_round, default_trace
from .metrics import make_metrics


def build_commons_harvest(**cfg) -> EnvSpec:
    """Builder: keyword config overrides -> a runnable ``EnvSpec``.

    Accepts a ``grid=(H, W)`` shorthand in addition to ``height`` / ``width``. All other keys are
    forwarded to ``HarvestConfig`` (e.g. ``n_agents``, ``policy``, ``zap_enabled``).
    """
    if "grid" in cfg:
        h, w = cfg.pop("grid")
        cfg.setdefault("height", h)
        cfg.setdefault("width", w)
    config = HarvestConfig(**cfg)
    return EnvSpec(
        name="commons_harvest",
        config=config,
        init_fn=make_init_fn(config),
        round_fn=make_round(config),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


__all__ = [
    "HarvestConfig", "layout",
    "make_state", "make_init_fn", "make_round", "default_trace", "make_metrics",
    "build_commons_harvest",
]
