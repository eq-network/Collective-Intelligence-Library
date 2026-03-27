"""
Transform pipeline for the governed commons harvest experiment.

Each transform is GraphState -> GraphState, composed via core.category.sequential().
Governance is a swappable transform in position 1 of the pipeline.

The full step pipeline:
    vote -> harvest -> resource_dynamics -> reward -> learning -> prediction -> step_counter
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from core.graph import GraphState
from core.category import Transform, sequential

from .policies import (
    bandit_harvest_policy,
    cooperative_vote_policy,
    adversarial_vote_policy,
    resource_prediction_policy,
)


def _split_key(state: GraphState):
    """Consume and advance the RNG key in global_attrs."""
    key = state.global_attrs["rng_key"]
    key, subkey = jr.split(key)
    new_global = dict(state.global_attrs)
    new_global["rng_key"] = key
    return state.replace(global_attrs=new_global), subkey


# --- Vote Transforms (one per mechanism) -------------------------------------

def make_vote_transform(mechanism: str = "pdd",
                        representative_mask: jnp.ndarray = None) -> Transform:
    """Create a vote transform for the given governance mechanism.

    Args:
        mechanism: one of "pdd", "prd", "pld", "none"
        representative_mask: (n_agents,) bool, required for PRD
    """
    def vote_transform(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        lr = state.global_attrs["learning_rate"]
        harvest_levels = state.global_attrs["harvest_levels"]
        n_agents = state.node_types.shape[0]
        resource_health = state.global_attrs["resource_level"] / state.global_attrs["K"]

        vote_weights = state.node_attrs["vote_weights"]
        is_adversarial = state.node_types  # 0=coop, 1=adv

        # Compute both vote types for all agents, select via jnp.where
        keys = jr.split(key, n_agents)

        coop_votes = jax.vmap(
            lambda w, k: cooperative_vote_policy(w, resource_health, k, lr)
        )(vote_weights, keys)

        adv_votes = jax.vmap(
            lambda w, k: adversarial_vote_policy(w, resource_health, k, lr)
        )(vote_weights, keys)

        vote_actions = jnp.where(is_adversarial, adv_votes, coop_votes)
        vote_values = harvest_levels[vote_actions]

        # Aggregate by mechanism
        if mechanism == "pdd":
            quota = jnp.median(vote_values)
        elif mechanism == "prd":
            mask = representative_mask.astype(jnp.float32)
            quota = jnp.sum(vote_values * mask) / (jnp.sum(mask) + 1e-8)
        elif mechanism == "pld":
            pred_scores = state.node_attrs["prediction_scores"]
            weights = pred_scores / (jnp.sum(pred_scores) + 1e-8)
            quota = jnp.sum(vote_values * weights)
        else:  # "none"
            quota = harvest_levels[-1]

        new_global = dict(state.global_attrs)
        new_global["quota"] = quota
        return state.replace(global_attrs=new_global)

    return vote_transform


# --- Harvest Transform --------------------------------------------------------

def harvest_transform(state: GraphState) -> GraphState:
    """Agents select harvest levels, capped by quota and resource availability."""
    state, key = _split_key(state)
    lr = state.global_attrs["learning_rate"]
    harvest_levels = state.global_attrs["harvest_levels"]
    n_agents = state.node_types.shape[0]
    resource_health = state.global_attrs["resource_level"] / state.global_attrs["K"]
    quota = state.global_attrs["quota"]

    harvest_weights = state.node_attrs["harvest_weights"]
    keys = jr.split(key, n_agents)

    # All agents use the same bandit policy for harvest selection
    action_indices = jax.vmap(
        lambda w, k: bandit_harvest_policy(w, resource_health, k, lr)
    )(harvest_weights, keys)

    harvests = harvest_levels[action_indices]

    # Quota caps individual harvest
    capped_harvests = jnp.minimum(harvests, quota)

    # Scale if total exceeds available resource
    total_harvest = jnp.sum(capped_harvests)
    resource = state.global_attrs["resource_level"]
    scale = jnp.where(
        total_harvest > resource,
        resource / (total_harvest + 1e-8),
        1.0,
    )
    actual_harvests = capped_harvests * scale

    # Update resource
    new_resource = resource - jnp.sum(actual_harvests)
    new_resource = jnp.maximum(new_resource, 0.0)

    # Update state
    state = state.update_node_attrs("last_harvest", actual_harvests)
    state = state.update_node_attrs(
        "cumulative_harvest",
        state.node_attrs["cumulative_harvest"] + actual_harvests,
    )
    new_global = dict(state.global_attrs)
    new_global["resource_level"] = new_resource
    return state.replace(global_attrs=new_global)


# --- Resource Dynamics --------------------------------------------------------

def resource_dynamics_transform(state: GraphState) -> GraphState:
    """Logistic regrowth: R += r * R * (1 - R/K)."""
    R = state.global_attrs["resource_level"]
    K = state.global_attrs["K"]
    r = state.global_attrs["growth_rate"]

    regrowth = r * R * (1.0 - R / K)
    new_R = jnp.clip(R + regrowth, 0.0, K)

    new_global = dict(state.global_attrs)
    new_global["resource_level"] = new_R
    return state.replace(global_attrs=new_global)


# --- Reward Transform ---------------------------------------------------------

def reward_transform(state: GraphState) -> GraphState:
    """Compute rewards with Fehr-Schmidt inequality aversion for cooperators.

    Cooperative reward:
        harvest_i - alpha * max(0, harvest_i - mean)
                  - beta  * max(0, mean - harvest_i)
                  + sustainability_bonus * (R/K > 0.5)

    This fixes Bug #3: cooperative agents now genuinely prefer moderate,
    sustainable harvesting over max extraction.

    Adversarial reward: pure extraction (harvest_i).
    """
    harvests = state.node_attrs["last_harvest"]
    is_adversarial = state.node_types
    R = state.global_attrs["resource_level"]
    K = state.global_attrs["K"]

    mean_harvest = jnp.mean(harvests)
    resource_health = R / K

    # Fehr-Schmidt inequality aversion (alpha=0.5, beta=0.25)
    advantageous_inequality = jnp.maximum(0.0, harvests - mean_harvest)
    disadvantageous_inequality = jnp.maximum(0.0, mean_harvest - harvests)
    sustainability_bonus = 0.5 * (resource_health > 0.5).astype(jnp.float32)

    cooperative_reward = (
        harvests
        - 0.5 * advantageous_inequality
        - 0.25 * disadvantageous_inequality
        + sustainability_bonus
    )
    adversarial_reward = harvests

    rewards = jnp.where(is_adversarial, adversarial_reward, cooperative_reward)
    return state.update_node_attrs("rewards", rewards)


# --- Learning Transform -------------------------------------------------------

def learning_transform(state: GraphState) -> GraphState:
    """Multiplicative weights update on harvest and vote weights.

    Harvest weights: updated based on reward signal.
    Vote weights: updated based on resource health feedback —
    good resource health reinforces the voted action.
    """
    rewards = state.node_attrs["rewards"]
    last_harvest = state.node_attrs["last_harvest"]
    harvest_levels = state.global_attrs["harvest_levels"]
    n_levels = state.global_attrs["n_harvest_levels"]
    resource_health = state.global_attrs["resource_level"] / state.global_attrs["K"]

    # Find which action index each agent effectively took
    # (closest harvest level to actual harvest)
    action_indices = jnp.argmin(
        jnp.abs(last_harvest[:, None] - harvest_levels[None, :]),
        axis=1,
    )
    action_onehot = jax.nn.one_hot(action_indices, n_levels)

    # Update harvest weights: reinforce chosen action proportional to reward
    harvest_weights = state.node_attrs["harvest_weights"]
    harvest_update = action_onehot * rewards[:, None]
    new_harvest_weights = harvest_weights + harvest_update

    # Update vote weights: reinforce based on resource health
    # Good health (>0.5) = positive signal, poor health = negative
    vote_weights = state.node_attrs["vote_weights"]
    health_signal = resource_health - 0.5  # centered around 0
    vote_update = action_onehot * health_signal
    new_vote_weights = vote_weights + vote_update

    state = state.update_node_attrs("harvest_weights", new_harvest_weights)
    state = state.update_node_attrs("vote_weights", new_vote_weights)
    return state


# --- Prediction Transform -----------------------------------------------------

def prediction_transform(state: GraphState) -> GraphState:
    """Agents predict resource direction; scores updated via Brier score + EMA."""
    state, key = _split_key(state)
    n_agents = state.node_types.shape[0]
    R = state.global_attrs["resource_level"]
    K = state.global_attrs["K"]
    r = state.global_attrs["growth_rate"]
    decay = state.global_attrs["prediction_decay"]

    last_harvest = state.node_attrs["last_harvest"]
    total_harvest = jnp.sum(last_harvest)

    keys = jr.split(key, n_agents)
    predictions = jax.vmap(
        lambda k: resource_prediction_policy(R, K, r, total_harvest, k)
    )(keys)

    # Actual outcome: did resource increase this step?
    # (Compare current R to what it was before regrowth, approximated)
    regrowth = r * R * (1.0 - R / K)
    resource_increasing = (regrowth > total_harvest).astype(jnp.float32)

    # Brier-style accuracy: 1 - (prediction - outcome)^2
    accuracy = 1.0 - (predictions - resource_increasing) ** 2

    # EMA update
    old_scores = state.node_attrs["prediction_scores"]
    new_scores = decay * old_scores + (1.0 - decay) * accuracy

    state = state.update_node_attrs("prediction_scores", new_scores)
    return state


# --- Step Counter -------------------------------------------------------------

def step_counter_transform(state: GraphState) -> GraphState:
    """Increment the step counter."""
    new_global = dict(state.global_attrs)
    new_global["step"] = state.global_attrs["step"] + 1
    return state.replace(global_attrs=new_global)


# --- Composition --------------------------------------------------------------

def make_step_transform(mechanism: str = "pdd",
                        representative_mask: jnp.ndarray = None) -> Transform:
    """Compose the full step pipeline for a given governance mechanism.

    The pipeline:
        vote -> harvest -> resource_dynamics -> reward -> learning -> prediction -> step_counter
    """
    return sequential(
        make_vote_transform(mechanism, representative_mask),
        harvest_transform,
        resource_dynamics_transform,
        reward_transform,
        learning_transform,
        prediction_transform,
        step_counter_transform,
    )
