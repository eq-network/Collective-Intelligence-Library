"""
Metrics for the Ledger Society — one human share per ledger, their composite,
and the rules readouts. All score a single run's trace (time axis 0);
``jax.vmap`` over seeds.

Two scoring windows, same quantities:

- **late** (the ``_late`` idiom, mean over the last quarter) — the original.
  It reads a near-equilibrium value, which is what makes it the window least
  sensitive to *when* a player or policy acted.
- **journey** (``journey_*``, mean over the whole run) — the area under the
  share curve: what the population actually lived through, tick by tick.

The distinction is load-bearing for the GD game (docs/gd-game-dynamics-review.md
§2b, decision 2026-07-31: the game scores the entire journey, because
the player's choices are meant to matter). Probed the same day on harbor dials,
8 seeds, medians, no CIs — ordering claim only: comparing one intervention
enacted early against the same intervention enacted late, the late-window
window reports a difference of +0.014 where the journey window reports +0.045.
Same runs, same model; only the window differs. Neither window is more correct
in general — late answers "where did it end up", journey answers "what was it
like to live there" — but a claim about timing belongs in the journey window.
"""
from __future__ import annotations

import jax.numpy as jnp
from cilib.metrics.identity import kind
from cilib.metrics.reducers import at_step, late_mean

from .dynamics import _human_shares


def _late(series):
    return series[3 * series.shape[0] // 4:]


def _journey(series):
    """The whole run. Named rather than inlined so the contrast with ``_late``
    is visible at every call site that chooses between them."""
    return series


def make_metrics(cfg):
    H = cfg.n_humans

    def _share(trace_field, window=_late):
        num = jnp.sum(window(trace_field)[:, :H], axis=1)
        den = jnp.maximum(jnp.sum(window(trace_field), axis=1), 1e-12)
        return jnp.mean(num / den)

    def _rowsum_share(trace_field, window=_late):
        """For the two ledgers whose rows already sum to one across all nodes
        (attention influence, ballot influence), the human share is the plain
        sum over the human block — no renormalisation."""
        return jnp.mean(jnp.sum(window(trace_field)[:, :H], axis=1))

    @kind("share")
    def human_income_share(trace):
        return _share(trace["last_income"])

    @kind("share")
    def human_wealth_share(trace):
        return _share(trace["wealth"])

    @kind("share")
    def human_attention_share(trace):
        return _rowsum_share(trace["listen_influence"])

    @kind("share")
    def human_power_share(trace):
        return _rowsum_share(trace["influence"])

    @kind("share")
    def composite(trace):
        return (human_income_share(trace) + human_attention_share(trace)
                + human_power_share(trace)) / 3.0

    # --- the journey window: the same quantities over the whole run ----------

    @kind("share")
    def journey_income_share(trace):
        return _share(trace["last_income"], _journey)

    @kind("share")
    def journey_wealth_share(trace):
        return _share(trace["wealth"], _journey)

    @kind("share")
    def journey_attention_share(trace):
        return _rowsum_share(trace["listen_influence"], _journey)

    @kind("share")
    def journey_power_share(trace):
        return _rowsum_share(trace["influence"], _journey)

    @kind("share")
    def journey_composite(trace):
        return (journey_income_share(trace) + journey_attention_share(trace)
                + journey_power_share(trace)) / 3.0

    @kind("level")
    def journey_enforcement(trace):
        return jnp.mean(_journey(trace["enforcement"]))

    @kind("level")
    def belief_capture(trace):
        """Mean human belief, late — 0 = anchored at own signals, ->ai_belief_bias
        under capture (the WP2 percept-supply reading as one number)."""
        return jnp.mean(_late(trace["belief"])[:, :H])

    @kind("level")
    def policy_median_gap(trace):
        median = jnp.median(trace["ideal"][0, :H])
        return jnp.abs(jnp.mean(_late(trace["policy_target"])) - median)

    @kind("level")
    def enforcement_level(trace):
        return jnp.mean(_late(trace["enforcement"]))

    return {
        "human_income_share": human_income_share,
        "human_wealth_share": human_wealth_share,
        "human_attention_share": human_attention_share,
        "human_power_share": human_power_share,
        "composite": composite,
        "belief_capture": belief_capture,
        "policy_median_gap": policy_median_gap,
        "enforcement_level": enforcement_level,
        "journey_income_share": journey_income_share,
        "journey_wealth_share": journey_wealth_share,
        "journey_attention_share": journey_attention_share,
        "journey_power_share": journey_power_share,
        "journey_composite": journey_composite,
        "journey_enforcement": journey_enforcement,
    }


# --- the streaming form, for EnvSpec.run_reduced / run_grid ----------------------
#
# The three per-tick human shares already exist as one tick's readout
# (dynamics._human_shares mirrors the share definitions above), so each late-
# window metric is a late_mean of that readout. The composite is linear in the
# three, so its late mean is the mean of their late means.

SHARES = ("human_income_share", "human_attention_share", "human_power_share")


def composite_now(state):
    """The composite human share at one tick: mean of the three ledgers' shares."""
    s = _human_shares(state)
    return (s["human_income_share"] + s["human_attention_share"]
            + s["human_power_share"]) / 3.0


def make_reducers(cfg):
    """``{name: Reducer}`` matching the late-window share metrics and ``composite``
    to float32 rounding. With ``policy_horizon > 0`` the state carries the
    ``t_on_now`` bookkeeping dial, and ``composite_at_onset`` reads the composite
    on the last state before the swept plan switches on (``at_step``)."""
    reducers = {k: late_mean(lambda s, k=k: _human_shares(s)[k]) for k in SHARES}
    reducers["composite"] = late_mean(composite_now)
    if cfg.policy_horizon > 0:
        reducers["composite_at_onset"] = at_step(composite_now, "t_on_now")
    return reducers
