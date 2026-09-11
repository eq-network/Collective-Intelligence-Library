"""
Metrics for Value Contagion — one cell readout of the (S, P) phase table.

``human_origin_share`` is the culture readout: the fraction of HUMAN agents
holding human-origin culture, averaged over the last quarter of the run (the
``_late`` idiom — steadier than the final tick). It is deliberately one *axis*
of the deliverable, never a headline on its own: the register (cultural-register
§7) forbids a scalar "AI cultural share" summary because low share means two
different worlds depending on structure. Pair it with
``metrics.families.spectral.fiedler_partition_alignment`` (read from finals —
the network is static per run) to place a run in the 2×2:

    share high, alignment low   -> pluralism
    share low,  alignment low   -> assimilation (absorbed, together)
    share high, alignment high  -> parallel cultures (separate, intact)
    share low,  alignment high  -> displacement (separate AND losing) — the
                                   gradual-cultural-disempowerment corner

Each metric scores a *single run's* trace (time axis 0); ``jax.vmap`` it over
the seed axis for batches.
"""
from __future__ import annotations

import jax.numpy as jnp

from cilib.metrics.reducers import late_mean
from cilib.metrics.identity import kind


def _late(series):
    return series[3 * series.shape[0] // 4:]


def make_metrics(cfg):
    n_h = cfg.n_agents - cfg.n_ai

    @kind("share")
    def human_origin_share(trace):
        """Fraction of human agents holding human-origin culture, late-run mean."""
        return jnp.mean(1.0 - _late(trace["culture"])[:, :n_h])

    return {"human_origin_share": human_origin_share}


# --- the same suite, folded into the scan ----------------------------------------
# `make_metrics` scores a materialized (T, N) trajectory; at N=5000, T=2000, 16
# seeds that trace is ~1.28 GB, against a 0.45 MB sparse state. These reducers
# compute the identical numbers in an O(1) carry (see core/reduce.py). Both are
# kept: the trace path stays the default for small runs and for the web export,
# which needs per-node fields for playback.
#
# `human_origin_share` is a mean over the late window AND over humans. Every step
# contributes the same number of humans, so the per-step human mean can be folded
# with a plain equal-weight `late_mean` — no re-weighting needed.

def make_reducers(cfg):
    """``{name: Reducer}`` matching ``make_metrics`` value-for-value (to float32
    rounding — a streaming mean does not sum in the same order as ``jnp.mean``
    over a block). Pass to ``EnvSpec.run_reduced``."""
    n_h = cfg.n_agents - cfg.n_ai

    def human_origin_readout(state):
        return jnp.mean(1.0 - state.node_attrs["culture"][:n_h])

    return {"human_origin_share": late_mean(human_origin_readout)}
