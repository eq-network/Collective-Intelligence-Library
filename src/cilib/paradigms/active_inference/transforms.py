"""
The round steps of the active-inference paradigm, as pure transforms.

Each step is built by a ``make_*`` factory that closes over the static
``AIConfig`` (observation operator, priors, world schedule, hyperparameters) and
returns a pure function on ``GraphState``. They are vectorized over the agent (N)
and candidate (K) axes with ``jax.vmap`` / ``einsum`` so they run inside
``core.scan.run_scan`` and ``vmap`` over seeds.

The paper's per-round loop (observe -> infer -> [crisis] -> fuse -> forget) maps
onto these directly. Crucially, each "ablation flag" of the paper is the
presence/absence (or a config flag) of one of these steps: with
``track_logweights=False`` and K=1 you get the plain Gaussian baseline; turn on
prequential scoring + log-weight pooling and you get represented rivals.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random, vmap

from cilib.core.graph import GraphState

from .schema import AIConfig, PI, HVEC, LOGW, WOBS, TRUST
from .primitives import (
    fisher_deposit_weighted, predictive_logpdf_perchannel, info_mean, info_cov,
    row_stochastic, savage_dickey, zero_edge_prior,
)


def make_observe(cfg: AIConfig):
    """OBSERVE + INFER (+ optional prequential scoring, + optional reliability gate).

    Samples one observation per agent from the world ``phi(t)`` through ``H``,
    optionally down-weights surprising channels (Student-t gate), optionally
    scores each candidate's one-step predictive density into its log-weight, then
    deposits the (shared) Fisher information into every candidate.
    """
    H = cfg.H
    sigma_o = cfg.sigma_o
    m = cfg.m

    def observe(state: GraphState, t, key) -> GraphState:
        Pi = state.node_attrs[PI]       # (N,K,d,d)
        h = state.node_attrs[HVEC]      # (N,K,d)
        logw = state.node_attrs[LOGW]   # (N,K)
        wobs = state.node_attrs[WOBS]   # (N,m)
        N = Pi.shape[0]

        phi = cfg.phi_of_t(t)                       # (d,)
        mean_o = H @ phi                            # (m,)
        o = mean_o[None, :] + random.normal(key, (N, m)) * sigma_o   # (N,m)

        # Reliability gate: trust what does not keep surprising you (Student-t).
        if cfg.use_gate:
            def gate_one(Pi_i0, h_i0, o_i, wobs_i):
                mu = info_mean(Pi_i0, h_i0)                  # (d,)
                Sigma = info_cov(Pi_i0)                      # (d,d)
                predmean = H @ mu                            # (m,)
                predvar = jnp.sum((H @ Sigma) * H, axis=1) + sigma_o ** 2  # (m,)
                z2 = (o_i - predmean) ** 2 / predvar
                kappa = (cfg.nu + 1.0) / (cfg.nu + z2)       # (m,)
                return wobs_i * kappa
            w_eff = vmap(gate_one)(Pi[:, 0], h[:, 0], o, wobs)   # (N,m)
        else:
            w_eff = wobs

        # Prequential scoring: theory-laden, attention-weighted predictive evidence
        # into the log-weights (per-channel marginal density weighted by w_obs, so
        # camps reading the same world through different channels can support
        # different wirings).
        if cfg.track_logweights:
            def ll_agent(Pi_i, h_i, o_i, w_i):
                def per_cand(P, hv):
                    perch = predictive_logpdf_perchannel(P, hv, H, o_i, sigma_o)  # (m,)
                    return jnp.sum(w_i * perch)
                return vmap(per_cand)(Pi_i, h_i)                  # (K,)
            logw = logw + vmap(ll_agent)(Pi, h, o, wobs)         # (N,K)

        # Fisher deposit (shared likelihood) into every candidate, attention-weighted.
        def deposit_one(w_i, o_i):
            return fisher_deposit_weighted(H, o_i, sigma_o, w_i)
        J, j = vmap(deposit_one)(w_eff, o)                       # (N,d,d), (N,d)
        Pi = Pi + J[:, None, :, :]
        h = h + j[:, None, :]

        return state.replace(node_attrs={**state.node_attrs, PI: Pi, HVEC: h, LOGW: logw})

    return observe


def make_fuse(cfg: AIConfig):
    """FUSE: precision-weighted averaging across the trust graph (DeGroot in
    information form — a contraction), within frame, plus damped log-linear
    pooling of the log-weights (the hypothesis pool)."""
    m_pool = cfg.m_pool

    def fuse(state: GraphState) -> GraphState:
        Pi = state.node_attrs[PI]
        h = state.node_attrs[HVEC]
        logw = state.node_attrs[LOGW]
        W = row_stochastic(state.adj_matrices[TRUST])            # (N,N)

        Pi_f = jnp.einsum('ij,jkab->ikab', W, Pi)                # fuse within frame
        h_f = jnp.einsum('ij,jka->ika', W, h)
        logw_f = (1.0 - m_pool) * logw + m_pool * (W @ logw)     # damped hypothesis pool

        return state.replace(node_attrs={**state.node_attrs, PI: Pi_f, HVEC: h_f, LOGW: logw_f})

    return fuse


def make_forget(cfg: AIConfig):
    """FORGET: relax accumulated precision/potential toward each candidate's prior
    at rate ``rho`` (rho<1 lets the value tilt and the gate actually move the
    belief), and exponentially decay the log-weights toward uniform at rate
    ``logw_decay`` (so q(m) has finite memory and can reverse after a world flip)."""
    Pi0 = cfg.Pi0[None]    # (1,K,d,d)
    h0 = cfg.h0[None]      # (1,K,d)
    rho = cfg.rho
    lam = cfg.logw_decay
    forget_pi = rho < 1.0
    decay_w = lam < 1.0

    if not forget_pi and not decay_w:
        return lambda state: state

    def forget(state: GraphState) -> GraphState:
        na = dict(state.node_attrs)
        if forget_pi:
            na[PI] = Pi0 + rho * (na[PI] - Pi0)
            na[HVEC] = h0 + rho * (na[HVEC] - h0)
        if decay_w:
            # softmax is shift-invariant, so shrinking logw toward 0 = q(m) toward uniform
            na[LOGW] = lam * na[LOGW]
        return state.replace(node_attrs=na)

    return forget


def bmr_edge_score(cfg: AIConfig, state: GraphState, candidate: int,
                   edge: tuple) -> jax.Array:
    """REDUCE (structure learning): per-agent log Bayes factor ΔF for pruning a
    coupling ``edge = (u, v)`` from ``candidate``'s wiring, scored from the current
    posterior against the candidate's prior (closed-form Savage–Dickey).

    ΔF > 0 means the data are content to drop the edge (reduction favoured —
    "crisis" on that edge); ΔF < 0 means the data hold the coupling. Read-only;
    returns shape (N,). This is the "reduction points where the world disagrees"
    check from the paper's appendix.
    """
    u, v = edge
    Pi = state.node_attrs[PI][:, candidate]      # (N,d,d) posterior precision
    h = state.node_attrs[HVEC][:, candidate]     # (N,d)
    Pi0 = cfg.Pi0[candidate]                      # (d,d) prior
    h0 = cfg.h0[candidate]                        # (d,)
    Pi0_red = zero_edge_prior(Pi0, u, v)          # reduced prior: edge pinned to 0

    def score_one(Pi_i, h_i):
        delta_F, _, _ = savage_dickey(Pi_i, h_i, Pi0, h0, Pi0_red, h0)
        return delta_F

    return vmap(score_one)(Pi, h)                 # (N,)
