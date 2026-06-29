"""
Schema for the polycentric commons paradigm: config + GraphState factory.

Design (mirrors active_inference): static config lives in a frozen ``PolyConfig`` that transform
factories close over — never in ``global_attrs``. ``GraphState`` carries only the evolving arrays.
Only jnp-array ``global_attrs`` are dynamic across scan steps (graph.py partitions on type), so
the mutable scalars (resource_level, rng_key, step) and the running affiliation sum live there as
jnp arrays; everything static (rates, capacity, governance mode) lives in ``PolyConfig``.

Governance is one knob set: the *initial affiliation graph*, whether it is *frozen*, whether
*monitoring/enforcement* is on, and a *capture* bias. Atomized / monocentric / fixed_poly /
endogenous are all the same dynamics with different settings of these.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from .primitives import row_softmax

# --- node attr / adjacency / global keys (module constants) ------------------
HARVEST_W = "harvest_weights"
THETA = "theta"
AFF_LOGITS = "affiliation_logits"
VOTE_VALUE = "vote_value"
QUOTA = "quota"
LOCAL_HEALTH = "local_health"
LAST_HARVEST = "last_harvest"
CUM_HARVEST = "cumulative_harvest"
REWARDS = "rewards"
FIT = "fit"
AFFIL = "affiliation"          # adj_matrix key
RESOURCE = "resource_level"
RNG = "rng_key"
STEP = "step"
AFFIL_SUM = "affiliation_sum"

GOVERNANCE_MODES = ("atomized", "monocentric", "fixed_poly", "endogenous")


@dataclasses.dataclass(frozen=True)
class PolyConfig:
    """Static configuration closed over by the transforms (not stored in GraphState)."""
    n_agents: int = 16
    n_blocks: int = 4
    n_levels: int = 6
    governance: str = "endogenous"          # one of GOVERNANCE_MODES
    monitoring: bool = True                  # enforcement on? (off => no sanction; ablation knob)
    freeze_affiliation: bool = False         # if True, affiliation does not update (frozen controls)
    capture: float = 0.0                     # exogenous hub-pull on affiliation updates
    # resource dynamics (small commons so tragedy is the default attractor).
    # sustainable total regrowth ~ growth*K/4 must EXCEED Σθ (so enforced fit-quotas can be
    # sustainable) yet stay BELOW greedy extraction (so unenforced play collapses).
    K_cap: float = 160.0
    growth_rate: float = 0.8
    init_resource: float = 120.0
    # behaviour
    learning_rate: float = 0.1
    affiliation_lr: float = 0.5
    alpha_local: float = 0.6                 # weight of local vs global health
    monitor_cost: float = 0.03               # per-agent cost of monitoring, debited from the pool
    sanction_strength: float = 1.5           # penalty per unit of over-quota harvest (if monitored)
    defect_prob: float = 0.1                 # prob an enforced agent still exceeds quota (indeterminism)
    # heterogeneity (the primary independent variable)
    heterogeneity: float = 1.5               # spread of sub-community ideal harvests
    theta_center: float = 1.5                # central sustainable ideal
    theta_noise: float = 0.15                # within-block idiosyncrasy

    @property
    def harvest_levels(self) -> jax.Array:
        return jnp.arange(self.n_levels, dtype=jnp.float32)

    @property
    def block_size(self) -> int:
        return self.n_agents // self.n_blocks


def block_assignment(cfg: PolyConfig) -> jax.Array:
    """Contiguous block id per agent (the latent sub-communities)."""
    return jnp.repeat(jnp.arange(cfg.n_blocks), cfg.block_size)


def hub_mask(cfg: PolyConfig) -> jax.Array:
    """Column bias target for capture: pull affiliation toward block-0 agents."""
    return (block_assignment(cfg) == 0).astype(jnp.float32)


def make_theta(cfg: PolyConfig, key) -> jax.Array:
    """Per-agent local sustainable ideal. Sub-communities differ by ``heterogeneity``; with
    heterogeneity=0 all blocks share one ideal (homogeneous baseline)."""
    blk = block_assignment(cfg)
    centered = (jnp.arange(cfg.n_blocks) - (cfg.n_blocks - 1) / 2.0)
    spread = centered / (max(cfg.n_blocks - 1, 1) / 2.0)        # in [-1, 1]
    block_ideal = cfg.theta_center + spread * cfg.heterogeneity  # (n_blocks,)
    theta = block_ideal[blk] + cfg.theta_noise * jr.normal(key, (cfg.n_agents,))
    return jnp.clip(theta, 0.0, cfg.n_levels - 1.0).astype(jnp.float32)


def _affiliation_init(cfg: PolyConfig, key) -> jax.Array:
    """Initial affiliation logits per governance mode (affiliation = row_softmax(logits))."""
    N = cfg.n_agents
    big = 8.0
    if cfg.governance == "atomized":
        return big * jnp.eye(N)                                   # ~identity → own vote
    if cfg.governance == "monocentric":
        return jnp.zeros((N, N))                                  # uniform → global mean
    if cfg.governance == "fixed_poly":
        blk = block_assignment(cfg)
        same = (blk[:, None] == blk[None, :]).astype(jnp.float32)
        return big * same                                        # block-diagonal → block mean
    # endogenous: small random logits, to be shaped by learning
    return 0.1 * jr.normal(key, (N, N))


def make_state(cfg: PolyConfig, key) -> GraphState:
    """Build the initial GraphState for one run/seed."""
    k_theta, k_aff = jr.split(key)
    N, L = cfg.n_agents, cfg.n_levels

    theta = make_theta(cfg, k_theta)
    logits = _affiliation_init(cfg, k_aff)
    affiliation = row_softmax(logits)

    node_types = jnp.zeros(N, dtype=jnp.int32)
    node_attrs = {
        HARVEST_W: jnp.ones((N, L), dtype=jnp.float32),
        THETA: theta,
        AFF_LOGITS: logits.astype(jnp.float32),
        VOTE_VALUE: jnp.zeros(N, dtype=jnp.float32),
        QUOTA: jnp.full((N,), cfg.theta_center, dtype=jnp.float32),
        LOCAL_HEALTH: jnp.full((N,), 0.75, dtype=jnp.float32),
        LAST_HARVEST: jnp.zeros(N, dtype=jnp.float32),
        CUM_HARVEST: jnp.zeros(N, dtype=jnp.float32),
        REWARDS: jnp.zeros(N, dtype=jnp.float32),
        FIT: jnp.zeros(N, dtype=jnp.float32),
    }
    adj_matrices = {AFFIL: affiliation.astype(jnp.float32)}
    global_attrs = {
        RESOURCE: jnp.array(cfg.init_resource, dtype=jnp.float32),
        RNG: key,
        STEP: jnp.array(0, dtype=jnp.int32),
        AFFIL_SUM: jnp.zeros((N, N), dtype=jnp.float32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices=adj_matrices,
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: PolyConfig):
    """``key -> GraphState`` for run_scan_batch (per-seed independent init)."""
    return lambda key: make_state(cfg, key)
