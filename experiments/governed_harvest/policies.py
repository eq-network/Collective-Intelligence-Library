"""
Pure policy functions for the governed commons harvest experiment.

All functions are designed for jax.vmap: they operate on single-agent
data and get vectorized across the agent dimension externally.

No Python branching — agent-type dispatch uses jnp.where in transforms.
"""
import jax
import jax.numpy as jnp
import jax.random as jr


def bandit_harvest_policy(harvest_weights, resource_health, key, learning_rate=0.1):
    """Sample a harvest action from softmax over weights.

    Args:
        harvest_weights: (n_levels,) bandit weights for this agent
        resource_health: () scalar R/K, unused here but available for extensions
        key: PRNG key
        learning_rate: temperature scaling

    Returns:
        action_idx: () integer index into harvest_levels
    """
    probs = jax.nn.softmax(harvest_weights * learning_rate)
    return jr.choice(key, a=harvest_weights.shape[0], p=probs)


def cooperative_vote_policy(vote_weights, resource_health, key, learning_rate=0.1):
    """Sustainability-biased vote. Bias toward low harvest when resource is low.

    The sustainability_bias creates a gradient from +2 (favor low harvest)
    to -2 (penalize high harvest), scaled by resource stress (1 - health).
    When the resource is healthy, the bias vanishes and weights dominate.
    """
    n = vote_weights.shape[0]
    sustainability_bias = jnp.linspace(2.0, -2.0, n)
    adjusted = vote_weights + sustainability_bias * (1.0 - resource_health)
    probs = jax.nn.softmax(adjusted * learning_rate)
    return jr.choice(key, a=n, p=probs)


def adversarial_vote_policy(vote_weights, resource_health, key, learning_rate=0.1):
    """Always vote for maximum harvest."""
    return jnp.array(vote_weights.shape[0] - 1, dtype=jnp.int32)


def resource_prediction_policy(resource_level, K, growth_rate, last_total_harvest, key):
    """Predict whether resource will increase next step.

    Uses a simple logistic regrowth model to estimate net change,
    then applies sigmoid to produce p(increase) in [0,1].
    Adds noise via Bernoulli sampling to avoid deterministic predictions.

    Returns:
        prediction: () float, 1.0 if predicting increase, 0.0 otherwise
    """
    regrowth = growth_rate * resource_level * (1.0 - resource_level / K)
    p_increase = jax.nn.sigmoid(regrowth - last_total_harvest)
    return jr.bernoulli(key, p=p_increase).astype(jnp.float32)
