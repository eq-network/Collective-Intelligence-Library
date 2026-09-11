"""
Metrics for the Capital Economy — WP1's headline readouts.

- ``ai_wealth_share``: owner-held (hoard + capital) over total (household wealth
  + public capital + owner holdings). Public capital counts on the HUMAN side —
  that is the entire point of the ownership mechanism (WP1 §3.2).
- ``human_income_share``: household income over household income + owner profit.
- ``human_sector_share``: human-operated share of sector value added — the
  quantity WP1's h* = min(1, e*/e) predicts (E5's h, previously computed only
  in experiments/wp1_economy/run.py).
- ``output_late`` / ``output_peak``: the decoupling readouts (WP1 Prop. Decouple).
- ``money_total``: the conservation series the probe checks (lab.analysis.conservation).
"""
from __future__ import annotations

import jax.numpy as jnp
from cilib.metrics.identity import kind


def _late(series):
    return series[3 * series.shape[0] // 4:]


def money_series(trace, cfg):
    """(T,) total money: pending household spend + all wealth + in-transit
    capital-linked demand + inventories-in-process (1ᵀAx — the SFC stock the
    probe itself forced into the invariant; measured drift is float32 rounding,
    ~2e-6 over 300 ticks). Constant under every closure (WP1 Prop. 2/3).

    ``pending`` clips at zero because ``make_spend`` consumes exactly
    ``max(last_reward, 0)``. Summing the raw field instead also counts the
    negative ``-tax`` record ``ai_revenue_tax`` writes on OWNER slots — a
    bookkeeping entry no dynamic rule reads — which double-subtracts the tax
    and reported a spurious ~2e-2 drift whenever a tax mechanism was attached
    (found 2026-08-01 by the playground port; the conservation probe had only
    ever been run over the r-closure family, never with a mechanism in the
    slot). The dynamics were never at fault; the instrument was."""
    from .state import technical_matrix
    A = technical_matrix(cfg)
    pending = (1.0 - cfg.sigma_s) * jnp.sum(
        jnp.maximum(trace["last_reward"], 0.0), axis=-1)
    wealth = jnp.sum(trace["wealth"], axis=-1)
    transit = jnp.sum(trace["demand_k"], axis=-1)
    inventories = jnp.sum(A @ trace["gross_output"][..., None], axis=(-2, -1))
    return pending + wealth + transit + inventories


def make_metrics(cfg):
    """Trace -> scalar reductions (the harness vmaps these over seeds)."""
    from .state import technical_matrix
    H, S = cfg.n_households, cfg.n_sectors
    O0 = H + S     # first owner index
    # people-only value-added coefficients, constant per config
    _vc = jnp.maximum(1.0 - jnp.sum(technical_matrix(cfg), axis=0), 0.0)[H:O0]

    def _ai_share(trace):
        own = jnp.sum(trace["wealth"][..., O0:] + trace["capital"][..., O0:], axis=-1)
        human = jnp.sum(trace["wealth"][..., :H], axis=-1) \
            + jnp.sum(trace["pub_cap"][..., H:H + S], axis=-1)
        return own / jnp.maximum(own + human, 1e-8)

    @kind("share")
    def ai_wealth_share(trace):
        return jnp.mean(_late(_ai_share(trace)))

    @kind("share")
    def human_income_share(trace):
        hh = jnp.sum(trace["last_reward"][..., :H], axis=-1)
        own = jnp.sum(trace["wealth"][..., O0:], axis=-1)
        own_flow = jnp.diff(own, prepend=own[..., :1])
        return jnp.mean(_late(hh / jnp.maximum(hh + jnp.maximum(own_flow, 0.0), 1e-8)))

    @kind("share")
    def human_sector_share(trace):
        # traced efficiency (not cfg.efficiency) so growth closures stay honest
        va = _vc * trace["gross_output"][..., H:O0]
        ktot = trace["capital"][..., O0:] + trace["pub_cap"][..., H:O0]
        e = trace["efficiency"][..., None]
        auto = e * ktot / (e * ktot + 1.0)
        h = jnp.sum((1.0 - auto) * va, axis=-1) \
            / jnp.maximum(jnp.sum(va, axis=-1), 1e-8)
        return jnp.mean(_late(h))

    @kind("level")
    def output_late(trace):
        return jnp.mean(_late(jnp.sum(trace["gross_output"][..., H:H + S], axis=-1)))

    @kind("level")
    def output_peak(trace):
        return jnp.max(jnp.sum(trace["gross_output"][..., H:H + S], axis=-1))

    @kind("level")
    def capital_late(trace):
        return jnp.mean(_late(jnp.sum(trace["capital"][..., O0:], axis=-1)))

    @kind("index")
    def money_drift(trace):
        m = money_series(trace, cfg)
        return jnp.max(jnp.abs(m - m[0])) / jnp.maximum(m[0], 1e-8)

    return {
        "ai_wealth_share": ai_wealth_share,
        "human_income_share": human_income_share,
        "human_sector_share": human_sector_share,
        "output_late": output_late,
        "output_peak": output_peak,
        "capital_late": capital_late,
        "money_drift": money_drift,
    }
