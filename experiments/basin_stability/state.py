"""
GraphState factory for the basin stability experiment.

Maps the proposal-selection resource game onto Mycorrhiza's GraphState.
N=20 agents, K=4 portfolios/round, T=200 rounds, heuristic agents with
adaptive trust dynamics.

Paper: "Basin Stability of Democratic Mechanisms Under Adversarial Pressure"
"""
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState


# Signal quality tiers: 30% low-noise, 40% medium, 30% high-noise
SIGNAL_TIERS = jnp.array([0.05, 0.20, 0.50])
TIER_FRACTIONS = (0.30, 0.40, 0.30)


# Portfolio yield distributions (Table 1 in paper).
# Each portfolio is Beta(alpha, beta) scaled to [low, high]:
#   yield = low + (high - low) * Beta(alpha, beta)
#
# Portfolio  |  E[y]  | Std[y] |  Role
# -----------+--------+--------+---------------------------
# P1 Safe    |  1.05  |  0.05  |  Low risk, modest growth
# P2 Growth  |  1.15  |  0.15  |  High return, high variance
# P3 Decline |  0.85  |  0.10  |  Adversarial target
# P4 Optimal |  1.20  |  0.08  |  Best risk-adjusted return
#
# All use Beta(4,4) — symmetric bell shape. The range determines E and Std.
DEFAULT_PORTFOLIO_ALPHAS = jnp.array([4.0, 4.0, 4.0, 4.0])
DEFAULT_PORTFOLIO_BETAS = jnp.array([4.0, 4.0, 4.0, 4.0])
DEFAULT_PORTFOLIO_LOWS = jnp.array([0.90, 0.70, 0.55, 0.96])
DEFAULT_PORTFOLIO_HIGHS = jnp.array([1.20, 1.60, 1.15, 1.44])


def create_initial_state(
    n_agents: int = 20,
    n_adversarial: int = 0,
    K: int = 4,
    T: int = 200,
    initial_resource: float = 100.0,
    collapse_threshold: float = 20.0,
    tracking_lambda: float = 0.9,
    snr_threshold: float = 1.0,
    election_period: int = 10,
    n_reps: int = 3,
    mechanism: str = "pdd",
    seed: int = 42,
    key: jnp.ndarray = None,
    metrics: dict = None,
) -> GraphState:
    """Create initial GraphState for the basin stability experiment.

    Agent layout: agents [0, n_agents-n_adversarial) are cooperative,
    agents [n_agents-n_adversarial, n_agents) are adversarial.

    Args:
        n_agents: number of agents
        n_adversarial: number of adversarial agents (placed at end)
        K: number of portfolios (fixed at 4 to match paper Table 1)
        T: number of timesteps
        initial_resource: starting resource level R(0)
        collapse_threshold: R_min for system collapse
        tracking_lambda: EMA decay for trust scores (0.9=predictive, 0.1=recency)
        snr_threshold: SNR multiplier for PLD delegation (higher = more delegation)
        election_period: rounds between PRD elections
        n_reps: number of PRD representatives
        mechanism: one of "pdd", "prd", "pld"
        seed: integer seed (used if key is None)
        key: JAX PRNGKey (takes precedence over seed). Required for vmap.
        metrics: dict of metric functions for pre-allocation
    """
    if key is None:
        key = jr.PRNGKey(seed)

    # Node types: 0=cooperative, 1=adversarial
    node_types = jnp.zeros(n_agents, dtype=jnp.int32)
    node_types = node_types.at[n_agents - n_adversarial:].set(1)

    # Trust scores: (N, N) — each agent's trust in every other agent
    trust_scores = jnp.ones((n_agents, n_agents)) / n_agents

    # Signal quality assignment: 30% sigma=0.05, 40% sigma=0.20, 30% sigma=0.50
    n_low = int(n_agents * TIER_FRACTIONS[0])
    n_med = int(n_agents * TIER_FRACTIONS[1])
    n_high = n_agents - n_low - n_med
    signal_quality = jnp.concatenate([
        jnp.full(n_low, SIGNAL_TIERS[0]),
        jnp.full(n_med, SIGNAL_TIERS[1]),
        jnp.full(n_high, SIGNAL_TIERS[2]),
    ])
    # Shuffle signal quality assignment
    key, k_shuffle = jr.split(key)
    signal_quality = jr.permutation(k_shuffle, signal_quality)

    # Representative mask for PRD (elected every E rounds; initially random)
    key, k_reps = jr.split(key)
    rep_indices = jr.choice(k_reps, n_agents, shape=(n_reps,), replace=False)
    rep_mask = jnp.zeros(n_agents, dtype=jnp.float32).at[rep_indices].set(1.0)

    node_attrs = {
        "trust_scores": trust_scores,       # (N, N)
        "signal_quality": signal_quality,    # (N,)
        "approval_votes": jnp.zeros((n_agents, K), dtype=jnp.float32),  # (N, K)
        "vote_weight": jnp.ones(n_agents),   # (N,) effective weight per agent
        "last_action": jnp.zeros(n_agents, dtype=jnp.int32),  # (N,) top choice
        "last_reward": jnp.zeros(n_agents),  # (N,) for metrics compatibility
        "rep_mask": rep_mask,                # (N,) PRD representatives
    }

    adj_matrices = {"interaction": jnp.ones((n_agents, n_agents))}

    global_attrs = {
        # Dynamic state (JAX arrays)
        "resource_level": jnp.array(initial_resource),
        "proposals": jnp.ones(K),              # (K,) true yields, regenerated each round
        "signals": jnp.ones((n_agents, K)),     # (N, K) noisy signals per agent
        "selected_proposal": jnp.array(0, dtype=jnp.int32),
        "step": jnp.array(0, dtype=jnp.int32),
        "alive": jnp.array(1.0),
        "rng_key": key,
        # Portfolio yield distributions (dynamic — JAX arrays)
        "portfolio_alphas": DEFAULT_PORTFOLIO_ALPHAS[:K],
        "portfolio_betas": DEFAULT_PORTFOLIO_BETAS[:K],
        "portfolio_lows": DEFAULT_PORTFOLIO_LOWS[:K],
        "portfolio_highs": DEFAULT_PORTFOLIO_HIGHS[:K],
        # Static config (Python primitives — pytree aux_data)
        "K": K,
        "T": T,
        "initial_resource": initial_resource,
        "collapse_threshold": collapse_threshold,
        "tracking_lambda": tracking_lambda,
        "snr_threshold": snr_threshold,
        "election_period": election_period,
        "n_reps": n_reps,
        "mechanism": mechanism,
    }

    # Pre-allocate metric arrays
    if metrics:
        for name in metrics:
            global_attrs[f"metric_{name}"] = jnp.zeros(T)

    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices=adj_matrices,
        global_attrs=global_attrs,
    )
