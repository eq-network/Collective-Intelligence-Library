"""
Metrics for Influence Exchange — the concentration curve and the wisdom readout.

All four score a single run's trace (time axis 0) with the ``_late`` idiom
(mean over the last quarter — steadier than the final tick); ``jax.vmap`` over
the seed axis for batches.

- ``human_influence_share``: total influence held by citizens. The headline —
  in the undefended amplified run it collapses; defenses restore it.
- ``influence_gini`` / ``centralization``: the A4 mainline indicators — how
  concentrated attention has become, over ALL nodes (organic preferential
  attachment concentrates within humans too; that's the pre-AI baseline).
- ``consensus_error``: |mean citizen opinion − truth| (truth = 0 by
  convention). Dispersed influence averages initial noise away (Golub-Jackson
  wisdom); influence concentrated on the biased AI reservoir drags consensus
  toward ``ai_bias`` — the wisdom condition breaking, as a number.
"""
from __future__ import annotations

import jax.numpy as jnp

from cilib.metrics.families.concentration import centralization_of, gini_of
from cilib.metrics.identity import kind


def _late(series):
    return series[3 * series.shape[0] // 4:]


def make_metrics(cfg):
    n_c = cfg.n_citizens

    @kind("share")
    def human_influence_share(trace):
        return jnp.mean(jnp.sum(_late(trace["influence"])[:, :n_c], axis=1))

    @kind("concentration")
    def influence_gini(trace):
        return gini_of(jnp.mean(_late(trace["influence"]), axis=0))

    @kind("concentration")
    def centralization(trace):
        return centralization_of(jnp.mean(_late(trace["influence"]), axis=0))

    @kind("level")
    def consensus_error(trace):
        return jnp.abs(jnp.mean(_late(trace["opinion"])[:, :n_c]))

    return {
        "human_influence_share": human_influence_share,
        "influence_gini": influence_gini,
        "centralization": centralization,
        "consensus_error": consensus_error,
    }
