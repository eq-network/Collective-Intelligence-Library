"""
Science / paradigm-shift scenarios — the "phlogiston world" family.

These factories build the initial GraphState AND the matching AIConfig (the
static constants closed over by the transforms). The Environment protocol only
returns a GraphState, so ``ScienceEnvironment`` stashes the cfg on itself; the
preset functions return ``(cfg, state)`` directly, which is what the agents and
tests consume.

The minimal faithful model (capture-the-mechanism, not bit-exact): agents hold
beliefs over ``d`` commitments; rival wirings are candidates that assert different
values for the commitments; the world ``phi(t)`` flips at ``t_shift`` so the
prequential evidence reorders the candidates — the Kuhn crossing. Theory-ladenness
is per-channel attention (``w_obs``): a camp that attends a channel its rival does
not accumulates evidence its rival cannot see.
"""
from __future__ import annotations

import jax.numpy as jnp

from core.graph import GraphState
from ..schema import AIConfig, make_state


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _diag_prior(prec: float, mu: jnp.ndarray):
    """Diagonal prior precision ``prec*I`` with potential ``h = Pi @ mu``."""
    d = mu.shape[0]
    Pi0 = prec * jnp.eye(d)
    return Pi0, Pi0 @ mu


def _complete_trust(n: int) -> jnp.ndarray:
    """Complete graph with self-loops (memory): ones(n, n). Row-normalized in fuse."""
    return jnp.ones((n, n))


# ---------------------------------------------------------------------------
# Scenario presets — each returns (cfg, state[, T]).
# ---------------------------------------------------------------------------

def two_camp_contraction(n_per_camp: int = 6, coupling: float = 0.8,
                         sigma_o: float = 0.5):
    """K=1, two camps with DIFFERENT wirings (camp A has a coupling edge, camp B
    does not). Plain precision-weighted fusion on a connected graph averages the
    structure away → the inter-camp wiring distance collapses (V5 Result 1)."""
    d, N = 2, 2 * n_per_camp
    H = jnp.eye(d)
    phi = jnp.ones(d)
    Pi0 = jnp.eye(d)[None]          # (1,d,d) — unused (rho=1, no forgetting anchor)
    h0 = jnp.zeros((1, d))
    cfg = AIConfig(H=H, Pi0=Pi0, h0=h0, phi_before=phi, phi_after=phi,
                   t_shift=10 ** 9, sigma_o=sigma_o, rho=1.0, m_pool=0.0,
                   track_logweights=False, use_gate=False)

    base = jnp.eye(d)
    PiA = base.at[0, 1].set(coupling).at[1, 0].set(coupling)   # camp A: an edge
    PiB = base                                                 # camp B: no edge
    Pi_init = jnp.concatenate([
        jnp.broadcast_to(PiA, (n_per_camp, d, d)),
        jnp.broadcast_to(PiB, (n_per_camp, d, d)),
    ])[:, None, :, :]                                          # (N,1,d,d)
    h_init = jnp.zeros((N, 1, d))
    logw = jnp.zeros((N, 1))
    wobs = jnp.ones((N, d))
    state = make_state(Pi_init, h_init, logw, wobs, _complete_trust(N))
    return cfg, state


def represented_rivals(n_per_camp: int = 6, sigma_o: float = 0.3,
                       m_pool: float = 0.05, rho: float = 0.5, prec: float = 4.0,
                       logw_decay: float = 0.85):
    """K=2, two camps in full contact reading the SAME world through DIFFERENT
    channels (theory-laden). Each camp's attention makes its own candidate predict
    better, so the damped hypothesis pool settles at a persistent gap rather than
    consensus — represented rivals hold divergence (V5 Result 2)."""
    d, K, N = 2, 2, 2 * n_per_camp
    mu0, mu1 = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    Pi0_0, h0_0 = _diag_prior(prec, mu0)
    Pi0_1, h0_1 = _diag_prior(prec, mu1)
    Pi0 = jnp.stack([Pi0_0, Pi0_1])           # (2,d,d)
    h0 = jnp.stack([h0_0, h0_1])              # (2,d)
    H = jnp.eye(d)
    phi = jnp.array([1.0, 1.0])               # both commitments hold in the world
    cfg = AIConfig(H=H, Pi0=Pi0, h0=h0, phi_before=phi, phi_after=phi,
                   t_shift=10 ** 9, sigma_o=sigma_o, rho=rho, m_pool=m_pool,
                   logw_decay=logw_decay, track_logweights=True, use_gate=False)

    Pi_init = jnp.broadcast_to(Pi0, (N, K, d, d))
    h_init = jnp.broadcast_to(h0, (N, K, d))
    # camp A attends channel 0 (favours cand0); camp B attends channel 1 (cand1)
    wobs = jnp.concatenate([
        jnp.broadcast_to(jnp.array([1.0, 0.0]), (n_per_camp, d)),
        jnp.broadcast_to(jnp.array([0.0, 1.0]), (n_per_camp, d)),
    ])
    logw = jnp.zeros((N, K))                   # uniform start; evidence does the work
    state = make_state(Pi_init, h_init, logw, wobs, _complete_trust(N))
    return cfg, state


def kuhn_cycle(n: int = 12, T: int = 120, sigma_o: float = 0.25,
               m_pool: float = 0.1, rho: float = 0.5, prec: float = 4.0,
               logw_decay: float = 0.9):
    """K=2 population, world flips at t_shift. From a uniform start the population
    commits to the incumbent; after the flip q(rival) climbs and the entropy of
    q(m) peaks (crisis as a torn state) before resolving — normal science, crisis,
    revolution, new normal (V5 Result 2 / Fig 4). Returns (cfg, state, T)."""
    d, K = 2, 2
    mu0, mu1 = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    Pi0_0, h0_0 = _diag_prior(prec, mu0)
    Pi0_1, h0_1 = _diag_prior(prec, mu1)
    Pi0 = jnp.stack([Pi0_0, Pi0_1])
    h0 = jnp.stack([h0_0, h0_1])
    H = jnp.eye(d)
    phi_before = jnp.array([1.0, 0.0])         # incumbent (cand0) reads it right
    phi_after = jnp.array([0.0, 1.0])          # rival (cand1) reads the flipped world
    cfg = AIConfig(H=H, Pi0=Pi0, h0=h0, phi_before=phi_before, phi_after=phi_after,
                   t_shift=T // 2, sigma_o=sigma_o, rho=rho, m_pool=m_pool,
                   logw_decay=logw_decay, track_logweights=True, use_gate=False)

    Pi_init = jnp.broadcast_to(Pi0, (n, K, d, d))
    h_init = jnp.broadcast_to(h0, (n, K, d))
    wobs = jnp.ones((n, d))                    # attend both channels
    logw = jnp.zeros((n, K))                   # uniform / normal science forms endogenously
    state = make_state(Pi_init, h_init, logw, wobs, _complete_trust(n))
    return cfg, state, T


def conflict_single_agent(use_gate: bool, t_conflict: int = 20, T: int = 60,
                          sigma_o: float = 0.3, rho: float = 0.85,
                          nu: float = 4.0, prec: float = 0.5):
    """One agent, K=1. A channel reads a consistent value, then flips to a
    contradictory one at t_conflict. With the reliability gate ON, the agent's
    posterior variance shows an INTERIOR maximum — doubt rises while data
    accumulate, then resolves — which no plain Gaussian (gate OFF) can produce
    (V5 Fig 8). Returns (cfg, state, T, t_conflict)."""
    d, K, N = 2, 1, 1
    H = jnp.eye(d)
    Pi0 = (prec * jnp.eye(d))[None]            # (1,d,d)
    h0 = jnp.zeros((1, d))
    phi_before = jnp.array([2.0, 0.0])         # channel 0 reads +2 ...
    phi_after = jnp.array([-2.0, 0.0])         # ... then contradicts at -2
    cfg = AIConfig(H=H, Pi0=Pi0, h0=h0, phi_before=phi_before, phi_after=phi_after,
                   t_shift=t_conflict, sigma_o=sigma_o, rho=rho, m_pool=0.0,
                   track_logweights=False, use_gate=use_gate, nu=nu)

    Pi_init = Pi0[None]                        # (N=1,K=1,d,d)
    h_init = jnp.zeros((N, K, d))
    logw = jnp.zeros((N, K))
    wobs = jnp.ones((N, d))
    state = make_state(Pi_init, h_init, logw, wobs, jnp.ones((N, N)))
    return cfg, state, T, t_conflict


class ScienceEnvironment:
    """Environment-protocol wrapper (``core.environment.Environment``).

    ``create_initial_state(**params)`` returns the GraphState; the matching
    ``AIConfig`` is stored on ``self.cfg`` (the transforms need it). ``preset`` is
    one of: 'two_camp_contraction', 'represented_rivals', 'kuhn_cycle'.
    """
    _PRESETS = {
        "two_camp_contraction": two_camp_contraction,
        "represented_rivals": represented_rivals,
        "kuhn_cycle": kuhn_cycle,
    }

    def __init__(self, preset: str = "kuhn_cycle"):
        if preset not in self._PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choices: {list(self._PRESETS)}")
        self.preset = preset
        self.cfg: AIConfig | None = None
        self.T: int | None = None

    def create_initial_state(self, **params) -> GraphState:
        out = self._PRESETS[self.preset](**params)
        if len(out) == 3:                      # kuhn_cycle returns (cfg, state, T)
            cfg, state, self.T = out
        else:
            cfg, state = out
        self.cfg = cfg
        return state
