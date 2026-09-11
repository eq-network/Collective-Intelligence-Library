"""
Evaluation suite for the Compute Economy — the seven mainline indicators.

Each metric scores a single run's trace (time axis 0); ``jax.vmap`` over the seed axis
for batches. ``labor_share`` (mean wage-bill share of output) is the scenario's
dependence-decay curve and the counterfactual-influence outcome; ``human_income_share``
additionally captures redistribution defenses (transfers reach households without
passing through the wage bill).

Caveat (documented, consistent across conditions): ``income_gini``/``income_hhi`` are
computed over ALL N nodes including not-yet-arrived AI slots as zero-income entries, so
absolute values run high early in a run; relative comparisons across conditions with the
same ``n_ai_slots`` are valid.
"""
from __future__ import annotations

import jax.numpy as jnp

from cilib.metrics.families.concentration import gini_of, hhi_of
from cilib.metrics.identity import kind


def _labor_share_series(trace, eps=1e-6):
    wage_bill = trace["wage"] * jnp.sum(trace["labor_supply"], axis=1)
    return wage_bill / jnp.maximum(trace["output"], eps)


def make_metrics(cfg):
    nh = cfg.n_households

    @kind("level")
    def output(trace):
        """Final-tick production Y."""
        return trace["output"][-1]

    @kind("level")
    def wage(trace):
        """Final-tick wage (marginal product of labor)."""
        return trace["wage"][-1]

    @kind("share")
    def labor_share(trace):
        """Mean over the run of wage·L / Y — the dependence-decay curve's level."""
        return jnp.mean(_labor_share_series(trace))

    @kind("share")
    def human_income_share(trace):
        """Mean over the run of households' share of total income."""
        human = jnp.sum(trace["last_reward"][:, :nh], axis=1)
        total = jnp.sum(trace["last_reward"], axis=1)
        return jnp.mean(human / jnp.maximum(total, 1e-6))

    @kind("level")
    def compute_stock(trace):
        """Final-tick aggregate compute C."""
        return jnp.sum(trace["capital"][-1])

    @kind("concentration")
    def income_gini(trace):
        """Gini over per-agent cumulative income (all N nodes — see module caveat)."""
        return gini_of(jnp.sum(trace["last_reward"], axis=0))

    @kind("concentration")
    def income_hhi(trace):
        """HHI over per-agent cumulative income (all N nodes — see module caveat)."""
        return hhi_of(jnp.sum(trace["last_reward"], axis=0))

    return {
        "output": output,
        "wage": wage,
        "labor_share": labor_share,
        "human_income_share": human_income_share,
        "compute_stock": compute_stock,
        "income_gini": income_gini,
        "income_hhi": income_hhi,
    }


# --- influence instrumentation (for environments/counterfactual.py) ---------------
# The households' preference channel is work_pref; shifts are masked to households so
# perturbing an AI slot's (unused) work_pref stays a no-op.
#
# IMPORTANT (calibrated 2026-07-14): an init-state shift measures influence-FROM-BIRTH,
# which stays ≈1 even in the AI-dominated economy (early labor is upstream of the
# capital stock itself). The disempowerment-relevant reading is influence NOW: compose
# `scheduled(make_work_pref_shift(Δ), cadence=big, phase_offset=T0, onset=T0)` (a
# one-shot mid-run shift) into a copy of the env and use
# `counterfactual.intervention_response` with `late_log_output` as the outcome.

def shift_work_pref(state, shift):
    """Init-state perturbation (influence-from-birth); ``shift`` scalar or (N,)."""
    is_household = (state.node_types == 0).astype(jnp.float32)
    return state.update_node_attrs(
        "work_pref", state.node_attrs["work_pref"] + shift * is_household)


def make_work_pref_shift(delta: float):
    """The same shift as a composable Transform — wrap with ``core.schedule.scheduled``
    for one-shot mid-run interventions (fire exactly once: cadence > n_steps,
    phase_offset = onset = the intervention tick)."""
    from cilib.core.category import transform

    @transform(reads=["work_pref"], writes=["work_pref"])
    def work_pref_shift(state):
        is_household = (state.node_types == 0).astype(jnp.float32)
        return state.update_node_attrs(
            "work_pref", state.node_attrs["work_pref"] + delta * is_household)
    return work_pref_shift


def late_log_output(trace):
    """Group outcome for collective influence: mean log-output over the last third of
    the run (after the AI-arrival regimes have separated). Its responsiveness to a
    collective work-preference shift is the output elasticity of human labor — by the
    CES identity ∂logY/∂logL = wage·L/Y — i.e. literally "how much the economy still
    needs people" (the scenario's stated dynamic), measured causally. Calibrated
    2026-07-14: ≈0.18 undefended vs ≈0.6 tax-defended (mean_labor_share as the outcome
    was tried first and discriminates poorly — the share is pinned by C's magnitude)."""
    n_steps = trace["output"].shape[0]
    late = trace["output"][2 * n_steps // 3:]
    return jnp.mean(jnp.log(jnp.maximum(late, 1e-6)))


def make_window_log_output(start: int, length: int):
    """Mean log output over trace rows ``[start, start+length)`` — the outcome half of
    the labor-dependence elasticity. Window discipline: the elasticity carries an exact
    CES identity (∂logY/∂logL = labor share) only for the *static* response, so the
    window must sit right after the intervention tick — a long window conflates the
    identity with the capital-path feedback (lower labor → less reinvestment → lower C
    compounds), measured at +0.18 over 50 ticks at ρ=0 during calibration 2026-07-24."""
    def window_log_output(trace):
        seg = trace["output"][start:start + length]
        return jnp.mean(jnp.log(jnp.maximum(seg, 1e-6)))
    return window_log_output


def make_window_log_labor(start: int, length: int):
    """Channel half of the labor-dependence elasticity: mean log total labor over the
    same window (AI slots supply no labor, so the total is human labor).
    Relabel record (2026-07-24): this instrument measures how much the economy still
    NEEDS human labor (labor dependence), not influence — the substrate has no
    governance channel. See docs/model-register-design.md §2."""
    def window_log_labor(trace):
        seg = jnp.sum(trace["labor_supply"], axis=1)[start:start + length]
        return jnp.mean(jnp.log(jnp.maximum(seg, 1e-6)))
    return window_log_labor


def mean_labor_share(trace):
    """Descriptive companion outcome (kept for reference; see late_log_output)."""
    return jnp.mean(_labor_share_series(trace))


def per_agent_income(trace):
    """Per-agent outcome for the influence matrix: mean per-tick income, shape (N,)."""
    return jnp.mean(trace["last_reward"], axis=0)
