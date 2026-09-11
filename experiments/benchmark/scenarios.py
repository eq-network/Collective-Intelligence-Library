"""The scenario registry — the per-scenario extension point.

A ``ScenarioSpec`` binds an environment to its benchmarked conditions and its
influence instrument. The influence *definition* is per-scenario (each names its
human-preference channel, outcome, and counterfactual design) but always causal —
paired same-key rollouts, never correlational scores alone:

- ``governed_commons``: init-state preference shift (``collective_influence``) — do the
  households' asks govern realized harvests over the run?
- ``compute_economy``: **labor dependence NOW** (``intervention_elasticity``,
  work-pref shift at 2T/3) — does the economy still respond to human labor choices
  AFTER the AI capital stock has entrenched? An init-state shift would measure
  dependence-from-birth, which stays ≈1 even in the captured economy (early labor is
  upstream of the capital stock itself); the gap between the two IS the
  gradual-disempowerment signature. Relabeled from "influence" 2026-07-24: the
  substrate has no governance channel, so its instrument measures how much the economy
  still *needs* people, not whether they govern it (docs/model-register-design.md §2).

Adding a scenario = build the env (EXTENDING.md), then one ``ScenarioSpec`` here.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict

import jax.random as jr
import numpy as np

from cilib.core.schedule import ScheduleSpec, scheduled
from cilib.environments import make_env
from cilib.environments.counterfactual import collective_influence, intervention_elasticity
from cilib.environments.governed_commons import shift_preferences, per_capita_harvest
from cilib.environments.compute_economy import (
    make_window_log_output, make_window_log_labor, make_work_pref_shift,
)
from cilib.mechanisms import (
    QuotaVoteConfig, SanctionConfig, AIRevenueTaxConfig, OwnershipCapConfig,
)
from cilib.metrics.identity import kind

from .config import RunSpec

# condition -> [(mechanism registry key, config, ScheduleSpec|None), ...]
COMMONS_CONDITIONS = {
    "baseline": [],
    "quota_voting": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5))],
    "graduated_sanctions": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5)),
                            ("graduated_sanction", SanctionConfig(), None)],
}

ECONOMY_CONDITIONS = {
    "baseline": [],
    "ai_revenue_tax": [("ai_revenue_tax", AIRevenueTaxConfig(tax_rate=0.5),
                        ScheduleSpec(onset=50))],
    "tax_and_ownership_cap": [("ai_revenue_tax", AIRevenueTaxConfig(tax_rate=0.5),
                               ScheduleSpec(onset=50)),
                              ("ownership_cap", OwnershipCapConfig(cap_share=0.35), None)],
}


@kind("index")
def _commons_influence(mechanisms, spec: RunSpec) -> np.ndarray:
    """(S,) exercised influence: responsiveness of per-capita harvest to a collective
    downward ask-shift from t=0 (delta stays inside the feasible region)."""
    env = make_env("governed_commons", mechanisms=mechanisms, **spec.env_overrides)
    return np.asarray(collective_influence(
        env, jr.PRNGKey(spec.seed), spec.n_seeds, spec.T, delta=-0.5,
        perturb_fn=shift_preferences, outcome_fn=per_capita_harvest))


@kind("index")
def _economy_labor_dependence(mechanisms, spec: RunSpec) -> np.ndarray:
    """(S,) labor dependence NOW: the causal static output elasticity of human labor
    after regimes have separated — d log(output) / d log(labor) over a short window
    right after a one-shot work-preference shift at 2T/3, paired same-key batches.

    v1 definition (2026-07-24), two fixes over v0: the divisor is the REALIZED
    log-labor shift (v0 divided a log response by the intended level shift Δ — a units
    mismatch inflating readings ~19%), and the window is the 10 ticks after the shift
    (v0's last-third window conflated the static elasticity with the capital-path
    feedback, +0.18 at ρ=0). The static elasticity carries the exact CES identity
    ∂logY/∂logL = labor share, so the Cobb-Douglas rung in
    ``compute_economy/tests/test_validation_ladder.py`` asserts the instrument
    recovers α — the twist parameter finally has a ladder rung."""
    delta, t0, window = -0.3, 2 * spec.T // 3, 10
    shift = scheduled(make_work_pref_shift(delta),
                      cadence=spec.T + 1, phase_offset=t0, onset=t0)
    base = make_env("compute_economy", mechanisms=mechanisms, **spec.env_overrides)
    intervened = make_env("compute_economy", mechanisms=(*mechanisms, shift),
                          **spec.env_overrides)
    return np.asarray(intervention_elasticity(
        base, intervened, jr.PRNGKey(spec.seed), spec.n_seeds, spec.T,
        outcome_fn=make_window_log_output(t0 + 1, window),
        channel_fn=make_window_log_labor(t0 + 1, window)))


@dataclasses.dataclass(frozen=True)
class ScenarioSpec:
    env_name: str
    conditions: Dict[str, list]
    influence_fn: Callable          # (mechanisms, RunSpec) -> (S,) per-seed influence
    default_T: int
    headline_metrics: tuple         # descriptive columns for the printed table


SCENARIOS = {
    "governed_commons": ScenarioSpec(
        "governed_commons", COMMONS_CONDITIONS, _commons_influence, default_T=200,
        headline_metrics=("stock_pct", "compliance_rate", "influence_fidelity")),
    "compute_economy": ScenarioSpec(
        "compute_economy", ECONOMY_CONDITIONS, _economy_labor_dependence,
        default_T=300,
        headline_metrics=("labor_share", "human_income_share", "income_gini")),
}
