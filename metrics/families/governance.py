"""
Governance metrics: delegation concentration, capture, voting patterns.

All functions are pure: GraphState -> JAX scalar.
Must NOT call float() — values stay as JAX arrays for JIT compatibility.
"""
import jax.numpy as jnp

from core.graph import GraphState


def delegation_gini(state: GraphState):
    """Gini coefficient of PLD delegation weights (trust column-sums).

    Measures concentration of voting power. Gini=0 means equal influence,
    Gini=1 means one agent holds all influence.
    """
    trust_scores = state.node_attrs["trust_scores"]
    weights = jnp.sum(trust_scores, axis=0)
    weights = weights / (jnp.sum(weights) + 1e-8)
    n = weights.shape[0]
    sorted_w = jnp.sort(weights)
    index = jnp.arange(1, n + 1, dtype=jnp.float32)
    return (2 * jnp.sum(index * sorted_w) / (n * jnp.sum(sorted_w))) - (n + 1) / n


def capture_rate(state: GraphState):
    """Fraction of PRD representatives that are adversarial."""
    rep_mask = state.node_attrs["rep_mask"]
    is_adv = state.node_types.astype(jnp.float32)
    n_reps = jnp.sum(rep_mask)
    n_adv_reps = jnp.sum(rep_mask * is_adv)
    return n_adv_reps / (n_reps + 1e-8)


def voting_entropy(state: GraphState):
    """Shannon entropy of the vote distribution across proposals.

    High entropy = dispersed votes, low entropy = consensus.
    """
    actions = state.node_attrs["last_action"]
    K = state.global_attrs["K"]
    votes_onehot = jnp.sum(jax_nn_one_hot(actions, K), axis=0)
    probs = votes_onehot / (jnp.sum(votes_onehot) + 1e-8)
    # Entropy with log2, masking zeros
    log_probs = jnp.where(probs > 1e-8, jnp.log2(probs), 0.0)
    return -jnp.sum(probs * log_probs)


def selected_utility(state: GraphState):
    """True utility of the proposal selected by the mechanism."""
    proposals = state.global_attrs["proposals"]
    selected = state.global_attrs["selected_proposal"]
    return proposals[selected]


def mean_trust(state: GraphState):
    """Mean pairwise trust score across all agents."""
    return jnp.mean(state.node_attrs["trust_scores"])


# Helper — avoid importing jax.nn at module level for the one_hot call
def jax_nn_one_hot(x, num_classes):
    """Inline one_hot to avoid import cycle."""
    return (x[..., None] == jnp.arange(num_classes)).astype(jnp.float32)


GOVERNANCE_METRICS = {
    "delegation_gini": delegation_gini,
    "capture_rate": capture_rate,
    "voting_entropy": voting_entropy,
    "selected_utility": selected_utility,
    "mean_trust": mean_trust,
}
