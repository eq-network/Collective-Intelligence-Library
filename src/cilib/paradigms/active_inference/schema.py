"""
State schema for the active-inference paradigm.

Two levels of graph, both carried in one GraphState (see README of engine/paradigms):

  - Outer graph: NODES ARE AGENTS. The social trust network is the adjacency
    ``adj_matrices["trust"]`` (its diagonal = memory, the self-loop).
  - Inner: each agent holds BELIEFS over ``d`` commitments, in information form
    ``(Pi, h)``, and may carry ``K`` candidate WIRINGS (paradigms) over those
    beliefs with a posterior ``q(m) = softmax(logw)`` over them.

node_attrs (leading axis is always N = number of agents):
    ai_Pi   : (N, K, d, d)  accumulated precision per agent per candidate wiring
    ai_h    : (N, K, d)     potential per agent per candidate
    ai_logw : (N, K)        log-weights over candidate wirings  ->  q(m)
    ai_wobs : (N, m)        per-agent channel attention (theory-ladenness / camp)

adj_matrices:
    trust   : (N, N)        social trust graph (include a positive diagonal = memory)

Scenario constants (the observation operator H, candidate priors, the world
schedule phi(t), and the hyperparameters) are NOT stored in GraphState — they are
static config closed over by the transforms (see AIConfig). GraphState carries only
the evolving arrays, so it stays a clean scan/vmap carry and global_attrs (static
pytree aux) never holds per-step data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp

from cilib.core.graph import GraphState

# node_attr / adjacency keys
PI = "ai_Pi"
HVEC = "ai_h"
LOGW = "ai_logw"
WOBS = "ai_wobs"
TRUST = "trust"


@dataclass(frozen=True)
class AIConfig:
    """Static scenario constants, closed over by the transforms (not in GraphState).

    H          : (m, d) observation operator (row k = the linear functional channel
                 k measures). Held constant — paradigms differ by prior topology,
                 never by the likelihood.
    Pi0, h0    : (K, d, d) / (K, d) prior precision/potential per candidate wiring.
                 The forgetting anchor and the BMR reference prior.
    phi_before : (d,) world truth before the regime shift.
    phi_after  : (d,) world truth after t_shift (set equal to phi_before for a
                 never-shifting control world).
    t_shift    : tick at which the world flips.
    sigma_o    : observation noise std (isotropic).
    rho        : forgetting rate in (0, 1]; 1.0 = no forgetting (precision only grows).
    m_pool     : damping of the log-weight (hypothesis) pool in [0, 1]; 0 = candidates
                 never communicate, 1 = full averaging (consensus is the only fixed
                 point). Small positive = damped pool with a persistent gap.
    logw_decay : exponential forgetting of the log-weights in (0, 1]; 1.0 = remember
                 all prequential evidence ever (q(m) cannot reverse in finite time),
                 < 1.0 = q(m) reflects recent predictive performance (so a world flip
                 can drive a crossing — the Kuhn cycle). Memory ~ 1/(1-logw_decay).
    nu         : Student-t dof for the reliability gate (large -> Gaussian, no gating).
    use_gate   : enable the Student-t reliability gate on observation channels.
    track_logweights : enable prequential scoring + log-weight dynamics (the
                 represented-rivals layer). Off => beliefs only (K is inert).
    """
    H: jax.Array
    Pi0: jax.Array
    h0: jax.Array
    phi_before: jax.Array
    phi_after: jax.Array
    t_shift: int
    sigma_o: float = 1.0
    rho: float = 1.0
    m_pool: float = 0.0
    logw_decay: float = 1.0
    nu: float = 4.0
    use_gate: bool = False
    track_logweights: bool = False

    @property
    def d(self) -> int:
        return self.H.shape[1]

    @property
    def m(self) -> int:
        return self.H.shape[0]

    @property
    def K(self) -> int:
        return self.Pi0.shape[0]

    def phi_of_t(self, t) -> jax.Array:
        """World truth at (traced) tick ``t``: flips from phi_before to phi_after
        at t_shift. Pure / jittable (uses jnp.where, not a Python branch)."""
        return jnp.where(t >= self.t_shift, self.phi_after, self.phi_before)


def make_state(Pi: jax.Array, h: jax.Array, logw: jax.Array,
               wobs: jax.Array, trust: jax.Array) -> GraphState:
    """Assemble an active-inference GraphState. N = Pi.shape[0]."""
    n = Pi.shape[0]
    return GraphState(
        node_types=jnp.zeros(n, dtype=jnp.int32),
        node_attrs={PI: Pi, HVEC: h, LOGW: logw, WOBS: wobs},
        adj_matrices={TRUST: trust},
        global_attrs={},
    )


def q_of_m(state: GraphState) -> jax.Array:
    """Posterior over candidate wirings per agent: softmax(logw) -> (N, K)."""
    from .primitives import softmax
    return softmax(state.node_attrs[LOGW], axis=-1)
