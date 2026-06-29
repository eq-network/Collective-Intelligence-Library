"""
Preset polycentric-commons configurations and run helpers.

A governance preset is just a setting of (initial affiliation, frozen?, monitoring?, capture).
All presets share the same dynamics, so differences in outcome are attributable to governance.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.scan import run_scan, run_scan_batch
from ..schema import (
    PolyConfig, make_state, make_init_fn,
    RESOURCE, LAST_HARVEST, FIT, QUOTA, AFFIL,
)
from ..transforms import make_round

# governance -> (freeze_affiliation, monitoring)
GOVERNANCE_PRESETS = {
    "atomized":    dict(freeze_affiliation=True,  monitoring=False),  # no institution, no enforcement
    "monocentric": dict(freeze_affiliation=True,  monitoring=True),   # one global quota, enforced
    "fixed_poly":  dict(freeze_affiliation=True,  monitoring=True),   # imposed blocks, enforced
    "endogenous":  dict(freeze_affiliation=False, monitoring=True),   # agents form institutions
}


def make_config(governance: str = "endogenous", heterogeneity: float = 1.5,
                monitoring: Optional[bool] = None, capture: float = 0.0,
                **overrides) -> PolyConfig:
    """Build a PolyConfig for a governance preset.

    ``monitoring`` overrides the preset default — set it False on an otherwise-governed config to
    run the *policer-ablation* (enforcement knockout). ``capture`` adds an exogenous hub-pull.
    """
    preset = GOVERNANCE_PRESETS[governance]
    kw = dict(governance=governance,
              freeze_affiliation=preset["freeze_affiliation"],
              monitoring=preset["monitoring"] if monitoring is None else monitoring,
              heterogeneity=heterogeneity,
              capture=capture)
    kw.update(overrides)
    return PolyConfig(**kw)


def default_trace(state):
    """Per-step readout: global stock, per-agent harvest, fit, and quota."""
    return {
        "resource": state.global_attrs[RESOURCE],
        "harvest": state.node_attrs[LAST_HARVEST],
        "fit": state.node_attrs[FIT],
        "quota": state.node_attrs[QUOTA],
    }


def commons(governance: str = "endogenous", heterogeneity: float = 1.5, T: int = 200, **overrides):
    """Return ``(cfg, init_fn, T)`` for a governance preset."""
    cfg = make_config(governance, heterogeneity, **overrides)
    return cfg, make_init_fn(cfg), T


def run(cfg: PolyConfig, key, T: int = 200, trace_fn: Callable = default_trace):
    """Single rollout. Returns ``(final_state, trace)``; trace fields have leading axis T."""
    state = make_state(cfg, key)
    return run_scan(make_round(cfg), state, T, key, trace_fn=trace_fn)


def run_batch(cfg: PolyConfig, key, n_seeds: int = 16, T: int = 200,
              trace_fn: Callable = default_trace):
    """Vmapped multi-seed sweep. Returns ``(final_states, traces)`` with leading axis (n_seeds,).
    Trace fields are shaped ``(n_seeds, T, ...)``."""
    keys = jr.split(key, n_seeds)
    return run_scan_batch(make_round(cfg), make_init_fn(cfg), T, keys, trace_fn=trace_fn)
