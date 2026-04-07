"""
GraphState factory for the basin stability experiment.

Maps the proposal-selection resource game onto Mycorrhiza's GraphState.
N=20 agents, K=4 proposals/round, T=200 rounds, linear Q-learning.

Paper: "Basin Stability of Democratic Mechanisms Under Adversarial Pressure"
"""
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState


# Signal quality tiers: 30% low-noise, 40% medium, 30% high-noise
SIGNAL_TIERS = jnp.array([0.05, 0.20, 0.50])
TIER_FRACTIONS = (0.30, 0.40, 0.30)


def create_initial_state(
    n_agents: int = 20,
    n_adversarial: int = 0,
    K: int = 4,
    T: int = 200,
    initial_resource: float = 100.0,
    collapse_threshold: float = 20.0,
    alpha: float = 0.01,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 100,
    trust_lambda: float = 0.9,
    election_period: int = 10,
    n_reps: int = 3,
    mechanism: str = "pdd",
    seed: int = 42,
) -> GraphState:
    """Create initial GraphState for the basin stability experiment.

    Agent layout: agents [0, n_agents-n_adversarial) are cooperative,
    agents [n_agents-n_adversarial, n_agents) are adversarial.

    State vector per agent: s_i = [R(t), u_hat_{i,1}, ..., u_hat_{i,K}] in R^{K+1}
    Q-function: Q(s, a) = w_a^T s + b_a

    For PLD, action space is K proposals + 1 delegate action = K+1 actions.
    For PDD/PRD, action space is K proposals.
    """
    key = jr.PRNGKey(seed)

    # Node types: 0=cooperative, 1=adversarial
    node_types = jnp.zeros(n_agents, dtype=jnp.int32)
    node_types = node_types.at[n_agents - n_adversarial:].set(1)

    # Action space: K for PDD/PRD, K+1 for PLD (extra delegate action)
    n_actions = K + 1  # always allocate K+1 for uniform shape; PDD/PRD ignore last
    state_dim = K + 1  # [R(t), u_hat_1, ..., u_hat_K]

    # Q-learning weights: (N, n_actions, state_dim) and biases: (N, n_actions)
    key, k1, k2 = jr.split(key, 3)
    q_weights = jr.normal(k1, (n_agents, n_actions, state_dim)) * 0.01
    q_bias = jnp.zeros((n_agents, n_actions))

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
        "q_weights": q_weights,           # (N, n_actions, state_dim)
        "q_bias": q_bias,                 # (N, n_actions)
        "trust_scores": trust_scores,     # (N, N)
        "signal_quality": signal_quality,  # (N,)
        "last_action": jnp.zeros(n_agents, dtype=jnp.int32),  # (N,)
        "last_reward": jnp.zeros(n_agents),                   # (N,)
        "rep_mask": rep_mask,             # (N,) — PRD representatives
    }

    adj_matrices = {"interaction": jnp.ones((n_agents, n_agents))}

    global_attrs = {
        "resource_level": jnp.array(initial_resource),
        "proposals": jnp.ones(K),              # (K,) — true utilities, regenerated each round
        "signals": jnp.ones((n_agents, K)),     # (N, K) — noisy signals per agent
        "selected_proposal": jnp.array(0, dtype=jnp.int32),
        "step": jnp.array(0, dtype=jnp.int32),
        "rng_key": key,
        # Static config
        "K": K,
        "T": T,
        "n_actions": n_actions,
        "state_dim": state_dim,
        "collapse_threshold": collapse_threshold,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_steps": epsilon_decay_steps,
        "trust_lambda": trust_lambda,
        "election_period": election_period,
        "n_reps": n_reps,
        "mechanism": mechanism,
    }

    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices=adj_matrices,
        global_attrs=global_attrs,
    )
