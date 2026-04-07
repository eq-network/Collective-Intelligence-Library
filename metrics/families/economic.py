"""
Economic metrics: resource health, welfare, inequality.

All functions are pure: GraphState -> JAX scalar.
Must NOT call float() — values stay as JAX arrays for JIT compatibility.
"""
import jax.numpy as jnp

from core.graph import GraphState


def resource_level(state: GraphState):
    """Current resource level."""
    return state.global_attrs["resource_level"]


def resource_health(state: GraphState):
    """Resource as fraction of initial level (R / R_0)."""
    R = state.global_attrs["resource_level"]
    # Use 100.0 as default initial; override via custom metric if needed
    R0 = state.global_attrs.get("initial_resource", 100.0)
    return R / R0


def mean_welfare(state: GraphState):
    """Mean reward across all agents."""
    return jnp.mean(state.node_attrs["last_reward"])


def cooperative_welfare(state: GraphState):
    """Mean reward of cooperative (type-0) agents."""
    rewards = state.node_attrs["last_reward"]
    is_coop = (state.node_types == 0).astype(jnp.float32)
    return jnp.sum(rewards * is_coop) / (jnp.sum(is_coop) + 1e-8)


def adversarial_welfare(state: GraphState):
    """Mean reward of adversarial (type-1) agents."""
    rewards = state.node_attrs["last_reward"]
    is_adv = (state.node_types == 1).astype(jnp.float32)
    return jnp.sum(rewards * is_adv) / (jnp.sum(is_adv) + 1e-8)


def gini_rewards(state: GraphState):
    """Gini coefficient of agent rewards. 0=equal, 1=maximally unequal."""
    rewards = state.node_attrs["last_reward"]
    # Shift to non-negative for Gini computation
    shifted = rewards - jnp.min(rewards) + 1e-8
    sorted_r = jnp.sort(shifted)
    n = sorted_r.shape[0]
    index = jnp.arange(1, n + 1, dtype=jnp.float32)
    return (2 * jnp.sum(index * sorted_r) / (n * jnp.sum(sorted_r))) - (n + 1) / n


ECONOMIC_METRICS = {
    "resource_level": resource_level,
    "resource_health": resource_health,
    "mean_welfare": mean_welfare,
    "cooperative_welfare": cooperative_welfare,
    "adversarial_welfare": adversarial_welfare,
    "gini_rewards": gini_rewards,
}
