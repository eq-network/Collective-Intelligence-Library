"""
GovSim-standard evaluation for Commons Harvest, bound to the trajectory trace.

These score a *single run's* trace (time axis = 0); for a batch of runs, ``jax.vmap`` the metric
(or ``EnvSpec.evaluate``) over the leading seed axis. They reuse the representation-agnostic GovSim
helpers in ``engine/environments/commons_metrics.py`` so numbers stay comparable to the LLM-agent
cooperation literature (Piatti et al., GovSim 2024).
"""
from __future__ import annotations

import jax.numpy as jnp

from ..commons_metrics import gini


def make_metrics(cfg):
    collapse_threshold = 1.0  # commons "dead" when fewer than ~1 apple remains

    def survival_time(trace):
        """Steps the commons stays viable before the apple stock collapses (full horizon if never)."""
        apples = trace["apples_total"]                     # (T,)
        alive = apples > collapse_threshold
        first_dead = jnp.argmax(~alive)
        return jnp.where(jnp.all(alive), apples.shape[-1], first_dead)

    def total_gain(trace):
        """Total apples harvested by the group over the run."""
        return jnp.sum(trace["harvest"])

    def equality(trace):
        """1 - Gini over per-agent cumulative harvest (1.0 = perfectly equal)."""
        return 1.0 - gini(jnp.sum(trace["harvest"], axis=0))

    def final_apples(trace):
        """Apple stock at the end of the run."""
        return trace["apples_total"][-1]

    return {
        "survival_time": survival_time,
        "total_gain": total_gain,
        "equality": equality,
        "final_apples": final_apples,
    }
