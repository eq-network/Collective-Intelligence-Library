"""
Polycentric commons paradigm — tragedy-of-commons with endogenous institution formation.

Agents harvest a shared logistic resource; each agent has a heterogeneous local sustainable
ideal (``theta``). Institutions are an affiliation graph that aggregates agents' votes into
per-agent quotas; in the ``endogenous`` mode agents form that graph themselves by affiliating
with similar-condition peers, so the institutional partition is instantiated by the agents
(Conant-Ashby), not imposed. Governance presets — atomized / monocentric / fixed_poly /
endogenous (+ a monitoring=False ablation, + a capture dial) — share one dynamics, so outcome
differences are attributable to governance.

Measured offline (see ``engine.analysis``): effective information across coarse-grainings (does
an institutional meso-scale emerge and is it causally privileged?) AND fit (does the setpoint
match heterogeneous local conditions?). The headline is the (EI, Fit) gap vs heterogeneity.
"""
from .schema import (
    PolyConfig, make_state, make_init_fn, block_assignment, hub_mask, make_theta,
    GOVERNANCE_MODES,
    HARVEST_W, THETA, AFF_LOGITS, VOTE_VALUE, QUOTA, LOCAL_HEALTH,
    LAST_HARVEST, CUM_HARVEST, REWARDS, FIT, AFFIL, RESOURCE, RNG, STEP, AFFIL_SUM,
)
from .transforms import make_round
from .agents import PolycentricAgent
from .environments import make_config, commons, run, run_batch, default_trace, GOVERNANCE_PRESETS

__all__ = [
    "PolyConfig", "make_state", "make_init_fn", "make_round", "PolycentricAgent",
    "block_assignment", "hub_mask", "make_theta", "GOVERNANCE_MODES", "GOVERNANCE_PRESETS",
    "make_config", "commons", "run", "run_batch", "default_trace",
    # field keys
    "HARVEST_W", "THETA", "AFF_LOGITS", "VOTE_VALUE", "QUOTA", "LOCAL_HEALTH",
    "LAST_HARVEST", "CUM_HARVEST", "REWARDS", "FIT", "AFFIL", "RESOURCE", "RNG", "STEP", "AFFIL_SUM",
]
