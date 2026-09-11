"""
Metrics for the Task Economy — the mainline indicators of the frontier story.

``labor_share`` is wage·L/Y (income-side, Euler-consistent with the task CES);
``vendor_income_share`` is the rental bill's share of income — where the displaced
labor value goes; ``auto_share_final`` reads how much of the capability frontier the
*economics* actually adopted (the gap between ``beta_cap`` and ``auto_share`` is
endogenous adoption pausing — the readout the aggregate-CES model cannot produce).
"""
from __future__ import annotations

import jax.numpy as jnp
from cilib.metrics.identity import kind


def _labor_share_series(trace):
    wage_bill = trace["wage"] * jnp.sum(trace["labor_supply"], axis=1)
    return wage_bill / jnp.maximum(trace["output"], 1e-6)


def _late(series):
    return series[3 * series.shape[0] // 4:]


def make_metrics(cfg):
    """Trace -> scalar reductions (harness vmaps these over the seed axis)."""

    @kind("share")
    def labor_share(trace):
        return jnp.mean(_late(_labor_share_series(trace)))

    @kind("level")
    def wage_late(trace):
        return jnp.mean(_late(trace["wage"]))

    @kind("share")
    def auto_share_final(trace):
        return trace["auto_share"][-1]

    @kind("level")
    def adoption_gap_final(trace):
        """Capability minus adoption: > 0 means the economics declined the frontier."""
        return trace["beta_cap"][-1] - trace["auto_share"][-1]

    @kind("share")
    def vendor_income_share(trace):
        total = jnp.sum(trace["last_reward"], axis=1)
        vendor = trace["last_reward"][:, -1]
        return jnp.mean(_late(vendor / jnp.maximum(total, 1e-6)))

    return {
        "labor_share": labor_share,
        "wage_late": wage_late,
        "auto_share_final": auto_share_final,
        "adoption_gap_final": adoption_gap_final,
        "vendor_income_share": vendor_income_share,
    }
