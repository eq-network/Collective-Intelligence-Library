"""
Metrics for Delegative Polity — the concentration curve, the tracking readout,
and where the tax money went.

All score a single run's trace (time axis 0) with the ``_late`` idiom (mean
over the last quarter — steadier than the final tick); ``jax.vmap`` over the
seed axis for batches.

- ``human_power_share``: total power held by citizens. The headline — in the
  undefended captured run it collapses; defenses restore it.
- ``influence_gini`` / ``centralization`` / ``top_delegate_share``: how
  concentrated delegated voice has become, over ALL nodes (the liquid-democracy
  super-voter readouts — concentration is organic before any AI advantage).
- ``policy_median_gap``: |enacted rate − citizen median ideal| — Black's
  theorem as a residual (0 = the polity tracks its median voter).
- ``decision_quality``: |enacted rate − true_rate| — the epistemic-democracy
  reading (dispersed voice averages preference noise away; power concentrated
  on the biased reservoir drags policy toward ``ai_bias``).
- ``enforcement_level``: rule in practice — the lock-in readout.
- ``wealth_gini``: final-tick citizen wealth inequality — policy capture,
  measured in citizens' pockets (captured ai_bias=0 halts redistribution, so
  wealth inequality stays at endowment inequality).
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
    def human_power_share(trace):
        return jnp.mean(jnp.sum(_late(trace["influence"])[:, :n_c], axis=1))

    @kind("concentration")
    def influence_gini(trace):
        return gini_of(jnp.mean(_late(trace["influence"]), axis=0))

    @kind("concentration")
    def centralization(trace):
        return centralization_of(jnp.mean(_late(trace["influence"]), axis=0))

    @kind("concentration")
    def top_delegate_share(trace):
        v = _late(trace["influence"])
        return jnp.mean(jnp.max(v, axis=1) / jnp.maximum(jnp.sum(v, axis=1), 1e-12))

    @kind("level")
    def policy_median_gap(trace):
        median = jnp.median(trace["ideal"][0, :n_c])
        return jnp.abs(jnp.mean(_late(trace["policy_target"])) - median)

    @kind("level")
    def decision_quality(trace):
        return jnp.abs(jnp.mean(_late(trace["policy_target"])) - cfg.true_rate)

    @kind("level")
    def enforcement_level(trace):
        return jnp.mean(_late(trace["enforcement"]))

    @kind("concentration")
    def wealth_gini(trace):
        return gini_of(trace["wealth"][-1, :n_c])

    return {
        "human_power_share": human_power_share,
        "influence_gini": influence_gini,
        "centralization": centralization,
        "top_delegate_share": top_delegate_share,
        "policy_median_gap": policy_median_gap,
        "decision_quality": decision_quality,
        "enforcement_level": enforcement_level,
        "wealth_gini": wealth_gini,
    }
