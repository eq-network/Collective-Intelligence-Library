"""
Transform pipeline for the polycentric commons paradigm.

Each transform is a pure ``GraphState -> GraphState`` built by a factory closing over a
``PolyConfig`` and decorated with ``@transform(reads, writes)`` declaring the GraphState
fields it touches. ``make_round`` hands the decorated steps to ``compile_pipeline``, which
derives execution order from those read/write sets (independent steps — e.g. local_health /
fit / reward — run as one parallel batch) and adapts the result to the ``run_scan``
``(state, t, key) -> state`` contract. Reuses the governed_harvest patterns (bandit harvest,
RNG threading via ``_split_key``, multiplicative-weights learning, logistic regrowth); the new
pieces are the per-agent quota via the affiliation graph, costly monitoring/sanctioning, the
fit measure, and the endogenous affiliation update.

The whole round:
    vote -> local_quota -> harvest -> resource_dynamics -> local_health -> fit
         -> reward -> learning -> [affiliation_update] -> accumulate -> step_counter
(affiliation_update is omitted when cfg.freeze_affiliation — the frozen-W controls).
All governance branching is on static cfg fields, so it resolves at trace-build (no lax.cond).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.category import transform
from cilib.core.pipeline import compile_pipeline
from cilib.core.scan import RoundFn

from .schema import (
    PolyConfig, hub_mask,
    HARVEST_W, THETA, AFF_LOGITS, VOTE_VALUE, QUOTA, LOCAL_HEALTH,
    LAST_HARVEST, CUM_HARVEST, REWARDS, FIT, AFFIL, RESOURCE, RNG, STEP, AFFIL_SUM,
)
from .primitives import row_softmax, aggregate, restraint_from_harvest


def _split_key(state: GraphState):
    """Consume and advance the RNG key in global_attrs (governed_harvest pattern)."""
    key = state.global_attrs[RNG]
    key, sub = jr.split(key)
    return state.update_global_attr(RNG, key), sub


# --- collective choice: each agent expresses its local sustainable ideal -----

def make_vote(cfg: PolyConfig):
    @transform(reads=[THETA], writes=[VOTE_VALUE])
    def vote(state: GraphState) -> GraphState:
        # truthful local ideal; the institution's job is to fit it, not to discover it
        return state.update_node_attrs(VOTE_VALUE, state.node_attrs[THETA])
    return vote


# --- institutional aggregation: per-agent quota via the affiliation graph -----

def make_local_quota(cfg: PolyConfig):
    @transform(reads=[AFFIL, VOTE_VALUE], writes=[QUOTA])
    def local_quota(state: GraphState) -> GraphState:
        W = state.adj_matrices[AFFIL]
        quota = aggregate(W, state.node_attrs[VOTE_VALUE])
        return state.update_node_attrs(QUOTA, quota)
    return local_quota


# --- harvest: greedy desire, capped at quota iff enforced --------------------

def make_harvest(cfg: PolyConfig):
    levels = cfg.harvest_levels
    L, N = cfg.n_levels, cfg.n_agents

    @transform(reads=[RNG, HARVEST_W, QUOTA, RESOURCE, CUM_HARVEST],
               writes=[RNG, LAST_HARVEST, CUM_HARVEST, RESOURCE])
    def harvest(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        hw = state.node_attrs[HARVEST_W]
        quota = state.node_attrs[QUOTA]
        keys = jr.split(key, N)

        def draw(w, k):
            probs = jax.nn.softmax(w * cfg.learning_rate)
            return jr.choice(k, a=L, p=probs)

        idx = jax.vmap(draw)(hw, keys)
        desired = levels[idx]

        if cfg.monitoring:
            state, key2 = _split_key(state)
            defect = jr.bernoulli(key2, p=cfg.defect_prob, shape=(N,))
            taken = jnp.where(defect, desired, jnp.minimum(desired, quota))
        else:
            taken = desired                          # no enforcement → greedy

        R = state.global_attrs[RESOURCE]
        total = jnp.sum(taken)
        scale = jnp.where(total > R, R / (total + 1e-8), 1.0)
        actual = taken * scale

        new_R = R - jnp.sum(actual)
        if cfg.monitoring:
            new_R = new_R - cfg.monitor_cost * N      # monitoring costs the commons
        new_R = jnp.maximum(new_R, 0.0)

        state = state.update_node_attrs(LAST_HARVEST, actual)
        state = state.update_node_attrs(CUM_HARVEST, state.node_attrs[CUM_HARVEST] + actual)
        return state.update_global_attr(RESOURCE, new_R)
    return harvest


# --- logistic regrowth -------------------------------------------------------

def make_resource_dynamics(cfg: PolyConfig):
    @transform(reads=[RESOURCE], writes=[RESOURCE])
    def resource_dynamics(state: GraphState) -> GraphState:
        R = state.global_attrs[RESOURCE]
        regrow = cfg.growth_rate * R * (1.0 - R / cfg.K_cap)
        return state.update_global_attr(RESOURCE, jnp.clip(R + regrow, 0.0, cfg.K_cap))
    return resource_dynamics


# --- local health: affiliation-weighted restraint + global stock -------------

def make_local_health(cfg: PolyConfig):
    max_level = float(cfg.n_levels - 1)

    @transform(reads=[AFFIL, LAST_HARVEST, RESOURCE], writes=[LOCAL_HEALTH])
    def local_health(state: GraphState) -> GraphState:
        W = state.adj_matrices[AFFIL]
        restraint = restraint_from_harvest(state.node_attrs[LAST_HARVEST], max_level)
        rh = state.global_attrs[RESOURCE] / cfg.K_cap
        health = cfg.alpha_local * aggregate(W, restraint) + (1.0 - cfg.alpha_local) * rh
        return state.update_node_attrs(LOCAL_HEALTH, health)
    return local_health


# --- fit: setpoint-to-local-condition match (negative local regret) ----------

def make_fit(cfg: PolyConfig):
    @transform(reads=[LAST_HARVEST, THETA], writes=[FIT])
    def fit(state: GraphState) -> GraphState:
        h = state.node_attrs[LAST_HARVEST]
        theta = state.node_attrs[THETA]
        return state.update_node_attrs(FIT, -jnp.abs(h - theta))
    return fit


# --- reward: extraction minus sanction for exceeding quota -------------------

def make_reward(cfg: PolyConfig):
    @transform(reads=[LAST_HARVEST, QUOTA, RESOURCE], writes=[REWARDS])
    def reward(state: GraphState) -> GraphState:
        h = state.node_attrs[LAST_HARVEST]
        quota = state.node_attrs[QUOTA]
        rh = state.global_attrs[RESOURCE] / cfg.K_cap
        if cfg.monitoring:
            sanction = cfg.sanction_strength * jnp.maximum(h - quota, 0.0)
        else:
            sanction = 0.0
        r = h - sanction + 0.5 * rh                  # extraction, penalty, sustainability bonus
        return state.update_node_attrs(REWARDS, r)
    return reward


# --- learning: multiplicative-weights on harvest action ----------------------

def make_learning(cfg: PolyConfig):
    levels = cfg.harvest_levels
    L = cfg.n_levels

    @transform(reads=[REWARDS, LAST_HARVEST, HARVEST_W], writes=[HARVEST_W])
    def learning(state: GraphState) -> GraphState:
        rewards = state.node_attrs[REWARDS]
        h = state.node_attrs[LAST_HARVEST]
        idx = jnp.argmin(jnp.abs(h[:, None] - levels[None, :]), axis=1)
        onehot = jax.nn.one_hot(idx, L)
        hw = state.node_attrs[HARVEST_W] + onehot * rewards[:, None] * cfg.learning_rate
        hw = hw - jnp.max(hw, axis=1, keepdims=True)   # softmax-invariant shift (bounded under scan)
        return state.update_node_attrs(HARVEST_W, hw)
    return learning


# --- endogenous institution formation: affiliate with similar local conditions

def make_affiliation_update(cfg: PolyConfig):
    hub = hub_mask(cfg)[None, :]

    @transform(reads=[AFF_LOGITS, THETA], writes=[AFF_LOGITS, AFFIL])
    def affiliation_update(state: GraphState) -> GraphState:
        logits = state.node_attrs[AFF_LOGITS]
        theta = state.node_attrs[THETA]
        diff = theta[:, None] - theta[None, :]
        sim = -(diff ** 2)                              # prefer agents with similar local ideal
        new_logits = logits + cfg.affiliation_lr * sim + cfg.capture * hub
        new_logits = new_logits - jnp.max(new_logits, axis=1, keepdims=True)
        W = row_softmax(new_logits)
        state = state.update_node_attrs(AFF_LOGITS, new_logits)
        return state.update_adj_matrix(AFFIL, W)
    return affiliation_update


def make_accumulate(cfg: PolyConfig):
    @transform(reads=[AFFIL_SUM, AFFIL], writes=[AFFIL_SUM])
    def accumulate(state: GraphState) -> GraphState:
        return state.update_global_attr(AFFIL_SUM, state.global_attrs[AFFIL_SUM] + state.adj_matrices[AFFIL])
    return accumulate


def make_step_counter(cfg: PolyConfig):
    @transform(reads=[STEP], writes=[STEP])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr(STEP, state.global_attrs[STEP] + 1)
    return step_counter


# --- composition -------------------------------------------------------------

def round_factories(cfg: PolyConfig):
    """The ordered step factories for one round (affiliation_update only when learnable)."""
    factories = [make_vote, make_local_quota, make_harvest, make_resource_dynamics,
                 make_local_health, make_fit, make_reward, make_learning]
    if not cfg.freeze_affiliation:
        factories.append(make_affiliation_update)
    factories += [make_accumulate, make_step_counter]
    return factories


def make_round(cfg: PolyConfig) -> RoundFn:
    """Compose the round via ``compile_pipeline`` (order derived from read/write sets) and
    adapt it to the run_scan ``(state, t, key) -> state`` contract."""
    pipeline = compile_pipeline([f(cfg) for f in round_factories(cfg)])

    def round_fn(state: GraphState, t, key) -> GraphState:
        state = state.update_global_attr(RNG, key)    # drive randomness from the scan's per-step key
        return pipeline(state)
    return round_fn
