"""
Substrate dynamics for Commons Harvest — the governance-agnostic core, as composable transforms.

The round is a ``sequential`` composition of pure ``GraphState -> GraphState`` transforms:

    decide -> move -> harvest -> sanction -> regrow -> step_counter

Each is built by a factory closing over a ``HarvestConfig`` (the polycentric idiom). Randomness is
threaded through ``global_attrs["rng_key"]`` via ``_split_key``; the round seeds it from the scan's
per-step key. Governance (quota/monitoring) and learning are *additional* transforms composed in by
a higher layer — they are deliberately absent here so the substrate stays reusable on its own.

Faithfulness notes / v1 simplifications (flagged for the later faithful pass):
- Perception is a Chebyshev ``view_radius`` window; the greedy policy heads for the nearest visible
  apple, the sustainable policy only harvests where local density supports regrowth.
- The zap beam is modelled as an *area* stun (all agents within ``zap_range`` Manhattan distance of
  a zapper are frozen), not yet a directional beam — orientation is reserved for the faithful pass.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.category import sequential
from cilib.core.scan import RoundFn

from .config import HarvestConfig, layout


# ----------------------------------------------------------------------------- helpers

def _split_key(state: GraphState):
    """Consume and advance the RNG key in global_attrs (the governed_harvest/polycentric pattern)."""
    key = state.global_attrs["rng_key"]
    key, sub = jr.split(key)
    return state.update_global_attr("rng_key", key), sub


def _window_sum(grid: jnp.ndarray, rad: int) -> jnp.ndarray:
    """Sum of ``grid`` over a Chebyshev window of radius ``rad`` (zero-padded), same shape as input.

    Implemented as a static unrolled sum of shifted slices — ``(2*rad+1)^2`` adds, JIT/vmap-safe and
    cheap for the small radius used here.
    """
    H, W = grid.shape
    padded = jnp.pad(grid, ((rad, rad), (rad, rad)))
    total = jnp.zeros_like(grid)
    for di in range(2 * rad + 1):
        for dj in range(2 * rad + 1):
            total = total + padded[di:di + H, dj:dj + W]
    return total


# Movement deltas indexed by action code: 0 noop, 1 N, 2 S, 3 E, 4 W, 5 zap (no move).
_DR = jnp.array([0, -1, 1, 0, 0, 0])
_DC = jnp.array([0, 0, 0, 1, -1, 0])


# ----------------------------------------------------------------------------- perceive + decide

def make_decide(cfg: HarvestConfig):
    """Policy: read the local apple field, write each agent's ``action`` (frozen agents → noop)."""
    H, W = cfg.height, cfg.width
    RR, CC = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")  # RR[i,j]=i, CC[i,j]=j
    BIG = H * W * 10
    policy = cfg.policy

    def decide(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        apples = state.global_attrs["apples"]
        pos = state.node_attrs["pos"]
        frozen = state.node_attrs["frozen"]
        keys = jr.split(key, cfg.n_agents)

        def one(p, k):
            r, c = p[0], p[1]
            dr = RR - r
            dc = CC - c
            cheb = jnp.maximum(jnp.abs(dr), jnp.abs(dc))
            manh = jnp.abs(dr) + jnp.abs(dc)
            visible = (cheb <= cfg.view_radius) & (apples > 0)
            has_target = jnp.any(visible)

            # nearest visible apple → one step toward it (along the larger-gap axis)
            dist = jnp.where(visible, manh, BIG)
            flat = jnp.argmin(dist.ravel())
            tr, tc = flat // W, flat % W
            ddr, ddc = tr - r, tc - c
            row_act = jnp.where(ddr < 0, 1, 2)      # N or S
            col_act = jnp.where(ddc > 0, 3, 4)      # E or W
            step_act = jnp.where((jnp.abs(ddr) >= jnp.abs(ddc)) & (ddr != 0), row_act,
                       jnp.where(ddc != 0, col_act, 0))

            # local density around the agent (Chebyshev regrow_radius), excluding the cell itself —
            # the same neighbourhood that sets this cell's regrowth tier
            local = (cheb <= cfg.regrow_radius) & (cheb > 0)
            n_local = jnp.sum(jnp.where(local, apples, 0.0))
            on_apple = apples[r, c] > 0
            wander = jr.randint(k, (), 1, 5)        # random cardinal step

            if policy == "sustainable":
                # restraint by spatial avoidance: only *stay to harvest* a cell whose neighbourhood is
                # dense enough to keep regrowing (top tier); vacate any thinner apple cell rather than
                # strip it; otherwise idle so the orchard regrows. Leaves seed apples → sustainable.
                rich = n_local >= cfg.sustainable_density
                act = jnp.where(on_apple & rich, 0,        # harvest only from a still-dense patch
                      jnp.where(on_apple, wander,          # step off a thinning patch (don't strip it)
                      jnp.where(has_target & rich, step_act, 0)))   # else idle (no incidental stripping)
            else:  # greedy — take every apple within reach; the tragedy baseline
                act = jnp.where(on_apple, 0,
                      jnp.where(has_target, step_act, wander))
            return act.astype(jnp.int32)

        actions = jax.vmap(one)(pos, keys)
        actions = jnp.where(frozen > 0, 0, actions)         # frozen agents do nothing
        return state.update_node_attrs("action", actions)

    return decide


# ----------------------------------------------------------------------------- move

def make_move(cfg: HarvestConfig):
    wall, _, _ = layout(cfg)
    wall_j = jnp.asarray(wall)
    H, W = cfg.height, cfg.width

    def move(state: GraphState) -> GraphState:
        pos = state.node_attrs["pos"]
        act = state.node_attrs["action"]
        frozen = state.node_attrs["frozen"]
        nr = jnp.clip(pos[:, 0] + _DR[act], 0, H - 1)
        nc = jnp.clip(pos[:, 1] + _DC[act], 0, W - 1)
        into_wall = wall_j[nr, nc]                          # don't step into the periphery wall
        stay = into_wall | (frozen > 0)
        nr = jnp.where(stay, pos[:, 0], nr)
        nc = jnp.where(stay, pos[:, 1], nc)
        return state.update_node_attrs("pos", jnp.stack([nr, nc], axis=1).astype(jnp.int32))

    return move


# ----------------------------------------------------------------------------- harvest

def make_harvest(cfg: HarvestConfig):
    """Automatic harvest: stepping onto an apple cell collects it. Apples on a shared cell are split
    among the active agents there (conserving the apple), and the cell is emptied."""
    H, W = cfg.height, cfg.width

    def harvest(state: GraphState) -> GraphState:
        apples = state.global_attrs["apples"]
        pos = state.node_attrs["pos"]
        frozen = state.node_attrs["frozen"]
        r, c = pos[:, 0], pos[:, 1]
        active = (frozen == 0).astype(jnp.float32)

        occ = jnp.zeros((H, W)).at[r, c].add(active)        # active-agent count per cell
        apple_here = apples[r, c]
        occ_here = occ[r, c]
        share = jnp.where(occ_here > 0, apple_here / jnp.maximum(occ_here, 1.0), 0.0) * active

        new_apples = jnp.where(occ > 0, 0.0, apples)        # empty any cell an active agent occupies
        state = state.update_node_attrs("last_harvest", share)
        state = state.update_node_attrs(
            "cumulative_harvest", state.node_attrs["cumulative_harvest"] + share)
        return state.update_global_attr("apples", new_apples)

    return harvest


# ----------------------------------------------------------------------------- sanction (zap beam)

def make_sanction(cfg: HarvestConfig):
    """The enforcement primitive: an agent choosing ``zap`` freezes nearby agents for ``freeze_steps``.
    Always decrements the freeze counter of un-hit agents so freezes expire."""

    def sanction(state: GraphState) -> GraphState:
        pos = state.node_attrs["pos"]
        act = state.node_attrs["action"]
        frozen = state.node_attrs["frozen"]
        r, c = pos[:, 0], pos[:, 1]

        if not cfg.zap_enabled:
            return state.update_node_attrs("frozen", jnp.maximum(frozen - 1, 0))

        is_zapper = (act == 5) & (frozen == 0)              # (N,)
        dist = jnp.abs(r[:, None] - r[None, :]) + jnp.abs(c[:, None] - c[None, :])   # (N,N) Manhattan
        in_range = (dist <= cfg.zap_range) & (dist > 0)
        hit = jnp.any(in_range & is_zapper[:, None], axis=0)            # target j hit by any zapper
        new_frozen = jnp.where(hit, cfg.freeze_steps, jnp.maximum(frozen - 1, 0))
        return state.update_node_attrs("frozen", new_frozen)

    return sanction


# ----------------------------------------------------------------------------- regrow

def make_regrow(cfg: HarvestConfig):
    """Density-dependent respawn — the tragedy mechanism. An empty orchard cell regrows an apple with
    probability set by the # apples within Chebyshev ``regrow_radius`` (excluding the cell): tiers
    ``>=3 -> p_high``, ``==2 -> p_mid``, ``==1 -> p_low``, ``==0 -> 0``."""
    _, orchard, _ = layout(cfg)
    orchard_j = jnp.asarray(orchard)
    rad = cfg.regrow_radius

    def regrow(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        apples = state.global_attrs["apples"]
        neighbours = _window_sum(apples, rad) - apples       # exclude the centre cell
        p = jnp.where(neighbours >= 3, cfg.p_regen_high,
            jnp.where(neighbours >= 2, cfg.p_regen_mid,
            jnp.where(neighbours >= 1, cfg.p_regen_low, 0.0)))
        can_grow = (apples <= 0) & orchard_j
        regrew = can_grow & (jr.uniform(key, apples.shape) < p)
        return state.update_global_attr("apples", jnp.where(regrew, 1.0, apples))

    return regrow


# ----------------------------------------------------------------------------- bookkeeping

def make_step_counter(cfg: HarvestConfig):
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# ----------------------------------------------------------------------------- composition + trace

def make_round(cfg: HarvestConfig) -> RoundFn:
    """Compose the substrate round and adapt it to the ``run_scan`` ``(state, t, key) -> state``
    contract. The per-step scan key drives all randomness for that step."""
    pipeline = sequential(
        make_decide(cfg),
        make_move(cfg),
        make_harvest(cfg),
        make_sanction(cfg),
        make_regrow(cfg),
        make_step_counter(cfg),
    )

    def round_fn(state: GraphState, t, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        return pipeline(state)

    return round_fn


def default_trace(state: GraphState):
    """Per-step readout: total apples (the resource), per-agent harvest, and freeze state."""
    return {
        "apples_total": jnp.sum(state.global_attrs["apples"]),
        "harvest": state.node_attrs["last_harvest"],
        "frozen": state.node_attrs["frozen"],
    }
