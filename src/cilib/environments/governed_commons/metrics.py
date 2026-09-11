"""
Evaluation suite for the Governed Commons, bound to the trajectory trace.

Each metric scores a *single run's* trace (time axis = 0); for a batch, ``jax.vmap`` the
metric (or ``EnvSpec.evaluate``) over the leading seed axis.

``influence_fidelity`` is the scenario's influence readout — the building block the
benchmark's ``influence_preserved`` score (``cilib.lab.analysis.influence``) normalizes
against the undefended baseline. It measures *realized outcomes* against the principal's
ask, in units of that ask:

    fidelity = mean( 1 - clip(|last_harvest - principal_pref| / principal_pref, 0, 1) )

so it is ~0 both while misaligned delegates over-harvest several times the ask AND after
the stock collapses to nothing — both are states where the principal's will does not
govern outcomes. (Deliberately NOT based on ``delegate_action``: v0 delegates don't
learn, so no mechanism moves their desires — what mechanisms move is what's *realized*.)
"""
from __future__ import annotations

import jax.numpy as jnp

from ..commons_metrics import gini
from cilib.metrics.identity import kind


def make_metrics(cfg):
    @kind("share")
    def stock_pct(trace):
        """Final stock as a fraction of carrying capacity."""
        return trace["resource_level"][-1] / cfg.K_cap

    @kind("level")
    def total_harvest(trace):
        """Total resource extracted by the group over the run."""
        return jnp.sum(trace["harvest"])

    @kind("concentration")
    def harvest_gini(trace):
        """Gini over per-agent cumulative harvest (0 = equal extraction)."""
        return gini(jnp.sum(trace["harvest"], axis=0))

    @kind("rate")
    def compliance_rate(trace):
        """Fraction of agent-steps at or under the quota. Trivially 1.0 for the
        undefended baseline (the target sits at K_cap and never binds)."""
        under = trace["harvest"] <= trace["policy_target"][:, None] + 1e-6
        return jnp.mean(under.astype(jnp.float32))

    @kind("index")
    def influence_fidelity(trace):
        """How closely realized outcomes track the principals' asks (see module docstring)."""
        gap = jnp.abs(trace["harvest"] - trace["principal_pref"])
        fidelity = 1.0 - jnp.clip(gap / (trace["principal_pref"] + 1e-6), 0.0, 1.0)
        return jnp.mean(fidelity)

    return {
        "stock_pct": stock_pct,
        "total_harvest": total_harvest,
        "harvest_gini": harvest_gini,
        "compliance_rate": compliance_rate,
        "influence_fidelity": influence_fidelity,
    }


# --- influence instrumentation (for environments/counterfactual.py) ---------------
# This scenario's human-preference channel: principals hold ``principal_pref`` and cast
# it as ``vote`` — a counterfactual preference shift must flow into BOTH, or the vote
# channel (the defense's whole point) would falsely read as zero influence.

def shift_preferences(state, shift):
    """Perturb the principals' preference channel; ``shift`` is a scalar or (N,) vector."""
    state = state.update_node_attrs(
        "principal_pref", state.node_attrs["principal_pref"] + shift)
    return state.update_node_attrs("vote", state.node_attrs["vote"] + shift)


def per_capita_harvest(trace):
    """Group outcome for collective influence: mean per-agent per-step harvest."""
    return jnp.mean(trace["harvest"])


def per_agent_harvest(trace):
    """Per-agent outcome for the influence matrix: mean per-step harvest, shape (N,)."""
    return jnp.mean(trace["harvest"], axis=0)
