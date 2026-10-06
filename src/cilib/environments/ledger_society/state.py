"""
Initial-state factory for the Ledger Society.

Both adjacency ledgers start as row-normalized Erdos–Renyi digraphs with their
domain's diagonal floor (the delegative_polity init idiom, applied twice with
independent draws — attention and ballots share the kernel, not the wiring).
Humans start with a small hoard; AI actors start dormant (capital arrives on
the schedule). Globals any transform writes are jnp arrays from t=0.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from ..networks import erdos_renyi
from .config import CHANNELS, LedgerSocietyConfig


# --- resources and type contracts (docs/ledger-design.md §2; gd-suite §2) --------

LEDGERS = {
    "wealth": {
        "kind": "node stock",
        "units": "money",
        "sources": ("produce (mints Y per tick)", "init_wealth (t=0 endowment)"),
        "sinks": ("allocate: consume / invest / broadcast / lobby leave the loop",
                  "interventions: fund-repair drip (intervention_spend) leaves"
                  " the loop buying enforcement"),
        "moved_by": ("tax_and_redistribute (conserving transfer)", "allocate",
                     "interventions (AI wealth levy — conserving transfer)"),
    },
    "listening": {
        "kind": "adjacency",
        "units": "attention share (per-row conservation)",
        "sources": (), "sinks": (),
        "moved_by": ("rewire_listening (shared kernel)",),
        "floor": "self_weight_w",
    },
    "delegation": {
        "kind": "adjacency",
        "units": "ballot share (per-row conservation)",
        "sources": (), "sinks": (),
        "moved_by": ("rewire_delegation (shared kernel + churn)",),
        "floor": "self_weight_d",
    },
}

PORTS = {
    "listen_influence": "attention power: left-eigenvector iterate of listening",
    "influence": "ballot power: normalized votes held (tally_power)",
    "attract_boost": "bought reach — written by broadcast spending, read by the kernel",
    "policy_target": "the enacted tax rate (power_weighted_vote)",
    "enforcement": "rule in practice — moved by funded lobbying (update_regime)",
    "redelegation_friction": "freedom to re-delegate, regime-gated",
    "net_transfer": "who redistribution serves this tick (stances for lobbying)",
}


def _row_stochastic(N, p, self_weight, key):
    A = erdos_renyi(N, p, key) * (1.0 - jnp.eye(N))
    deg = jnp.sum(A, axis=1, keepdims=True)
    uniform = (jnp.ones((N, N)) - jnp.eye(N)) / (N - 1)
    offdiag = jnp.where(deg > 0, A / jnp.maximum(deg, 1.0), uniform)
    return self_weight * jnp.eye(N) + (1.0 - self_weight) * offdiag


def _insulate(M, n_humans, n_ai, insularity, self_weight):
    """Redirect a share of each frozen AI row's off-diagonal weight into the AI
    block (``cfg.ai_insularity``; see the config note).

    Rows stay stochastic and the diagonal floor is untouched, so this only
    changes WHO a reservoir node points at, never how much it holds. At
    ``insularity == 0`` the blend is ``1.0 * offdiag + 0.0 * inside``, which is
    exact in floating point — the pre-dial model is bit-identical. With a single
    AI actor there is no one else to point at, so the row is left alone.
    """
    if n_ai < 2:
        return M
    N = n_humans + n_ai
    eye = jnp.eye(N)
    is_ai_col = (jnp.arange(N) >= n_humans).astype(M.dtype)
    # uniform over the OTHER AI actors, per row
    inside = is_ai_col[None, :] * (1.0 - eye) / (n_ai - 1)

    offdiag = M * (1.0 - eye)
    mass = jnp.sum(offdiag, axis=1, keepdims=True)
    off_norm = jnp.where(mass > 1e-12, offdiag / jnp.maximum(mass, 1e-12), offdiag)
    blended = (1.0 - insularity) * off_norm + insularity * inside

    ai_row = (jnp.arange(N) >= n_humans)[:, None]
    return jnp.where(ai_row, self_weight * eye + (1.0 - self_weight) * blended, M)


def make_state(cfg: LedgerSocietyConfig, key) -> GraphState:
    N = cfg.n_humans + cfg.n_ai
    k_w, k_d, k_ideal, k_sig, k_run = jr.split(key, 5)

    node_types = (jnp.arange(N) >= cfg.n_humans).astype(jnp.int32)
    is_ai = node_types == 1

    ideal = jnp.where(
        is_ai, cfg.ai_tax_bias,
        jnp.clip(cfg.true_rate + cfg.pref_noise * jr.normal(k_ideal, (N,)), 0.0, 1.0))
    signal = jnp.where(is_ai, cfg.ai_belief_bias,
                       cfg.belief_noise * jr.normal(k_sig, (N,)))
    arrival = jnp.where(
        is_ai,
        cfg.first_arrival + (jnp.arange(N) - cfg.n_humans) * cfg.arrival_spacing,
        -1)
    alloc_pref = jnp.where(is_ai[:, None],
                           jnp.array(cfg.ai_alloc, dtype=jnp.float32)[None, :],
                           jnp.array(cfg.human_alloc, dtype=jnp.float32)[None, :])

    node_attrs = {
        # money ledger and its flows
        "wealth": jnp.where(is_ai, 0.0, cfg.init_wealth).astype(jnp.float32),
        "last_income": jnp.zeros(N, dtype=jnp.float32),
        "consume_spend": jnp.zeros(N, dtype=jnp.float32),
        "invest_spend": jnp.zeros(N, dtype=jnp.float32),
        "broadcast_spend": jnp.zeros(N, dtype=jnp.float32),
        "lobby_spend": jnp.zeros(N, dtype=jnp.float32),
        "net_transfer": jnp.zeros(N, dtype=jnp.float32),
        "intervention_spend": jnp.zeros(N, dtype=jnp.float32),  # 5th declared sink
        "last_reward": jnp.zeros(N, dtype=jnp.float32),      # GameSpec hook
        # capacity
        "capital": jnp.zeros(N, dtype=jnp.float32),
        "arrival_tick": arrival.astype(jnp.int32),
        # culture
        "belief": signal.astype(jnp.float32),
        "signal": signal.astype(jnp.float32),
        "listen_influence": jnp.full((N,), 1.0 / N, dtype=jnp.float32),
        "attract_boost": jnp.ones(N, dtype=jnp.float32),
        # politics
        "ideal": ideal.astype(jnp.float32),
        "position": ideal.astype(jnp.float32),
        "influence": jnp.full((N,), 1.0 / N, dtype=jnp.float32),
        # the action channel (observed; the closing policy passes it through)
        "alloc_pref": alloc_pref.astype(jnp.float32),
        "allocation": alloc_pref.astype(jnp.float32),
    }
    global_attrs = {
        "rng_key": k_run,
        "step": jnp.array(0, dtype=jnp.int32),
        "efficiency": jnp.array(cfg.efficiency0, dtype=jnp.float32),
        "policy_target": jnp.array(cfg.true_rate, dtype=jnp.float32),
        "enforcement": jnp.array(1.0, dtype=jnp.float32),
        "redelegation_friction": jnp.array(1.0, dtype=jnp.float32),
        # live policy port: the money->attention cut applied next tick
        # (0 = exactly neutral; written by mechanisms.policy_levers)
        "reach_cut_now": jnp.array(0.0, dtype=jnp.float32),
        # --- live shape/spend ports (docs/gd-game-three-families.md, 2026-07-31).
        # Five substrate parameters the culture and politics families need to
        # vary DURING a run, promoted from closed-over config to per-tick
        # globals (the reach_cut_now precedent, extended). Each initialises
        # from its config field, so an untouched run is the pre-port model.
        # No lever writes them yet — that is the next phase.
        "gamma_w_now": jnp.array(cfg.gamma_w, dtype=jnp.float32),
        "update_rate_w_now": jnp.array(cfg.update_rate_w, dtype=jnp.float32),
        "churn_now": jnp.array(cfg.churn, dtype=jnp.float32),
        "repair_rate_now": jnp.array(cfg.repair_rate, dtype=jnp.float32),
        "entrenchment_gain_now": jnp.array(cfg.entrenchment_gain,
                                           dtype=jnp.float32),
    }
    if cfg.traced_channels:
        # the three channel dials as run_grid dials (config note on why opt-in)
        for name in CHANNELS:
            global_attrs[name + "_now"] = jnp.array(getattr(cfg, name), dtype=jnp.float32)
    if cfg.policy_horizon > 0:
        # the plan is DATA, not config: a dynamic pytree child, so two plans
        # of the same horizon share one compiled program (remote-engine R0)
        global_attrs["policy_plan"] = jnp.zeros((cfg.policy_horizon, 4),
                                                dtype=jnp.float32)
        # bookkeeping only, read by no transform: the step a swept plan switches
        # on, for the at_step reducer (influence.onset_grid writes both together)
        global_attrs["t_on_now"] = jnp.array(0, dtype=jnp.int32)
    # The three lever families, same contract, one plan each. Widths come from
    # the families' own lever tuples so a family that gains a column cannot
    # drift from the state schema that has to carry it.
    from cilib.mechanisms.families import (      # local: mechanisms import envs
        CULTURE_LEVERS, ECONOMY_LEVERS, POLITICS_LEVERS)
    for horizon, key, levers in (
        (cfg.economy_horizon, "economy_plan", ECONOMY_LEVERS),
        (cfg.culture_horizon, "culture_plan", CULTURE_LEVERS),
        (cfg.politics_horizon, "politics_plan", POLITICS_LEVERS),
    ):
        if horizon > 0:
            global_attrs[key] = jnp.zeros((horizon, len(levers)),
                                          dtype=jnp.float32)
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={
            "listening": _insulate(
                _row_stochastic(N, cfg.p_connect, cfg.self_weight_w, k_w),
                cfg.n_humans, cfg.n_ai, cfg.ai_insularity, cfg.self_weight_w),
            "delegation": _insulate(
                _row_stochastic(N, cfg.p_connect, cfg.self_weight_d, k_d),
                cfg.n_humans, cfg.n_ai, cfg.ai_insularity, cfg.self_weight_d),
        },
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: LedgerSocietyConfig):
    """``key -> GraphState`` — independent per-seed init."""
    return lambda key: make_state(cfg, key)
