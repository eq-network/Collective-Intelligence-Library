"""
Transform pipeline for the basin stability experiment.

Each transform is GraphState -> GraphState, composed via core.category.sequential().
The mechanism (PDD/PRD/PLD) is selected at composition time, not runtime.

Full step pipeline:
    proposal_gen -> voting -> aggregation -> resource_update ->
    reward -> q_learning -> trust_update -> election -> step_counter

Paper: multiplicative resource dynamics R(t+1) = R(t) * u_{k*}
       with K proposals drawn from U(0.80, 1.25) each round.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from core.graph import GraphState
from core.category import Transform, sequential, conditional

from .policies import (
    q_select_action,
    q_update,
    trust_update,
    delegate_target,
)


def _split_key(state: GraphState):
    """Consume and advance the RNG key in global_attrs."""
    key = state.global_attrs["rng_key"]
    key, subkey = jr.split(key)
    new_global = dict(state.global_attrs)
    new_global["rng_key"] = key
    return state.replace(global_attrs=new_global), subkey


def _get_epsilon(state: GraphState):
    """Compute current exploration rate with linear decay."""
    step = state.global_attrs["step"]
    eps_start = state.global_attrs["epsilon_start"]
    eps_end = state.global_attrs["epsilon_end"]
    decay_steps = state.global_attrs["epsilon_decay_steps"]
    progress = jnp.clip(step / decay_steps, 0.0, 1.0)
    return eps_start + (eps_end - eps_start) * progress


def _build_state_vec(resource_level, agent_signals):
    """Build state vector s_i = [R(t), u_hat_{i,1}, ..., u_hat_{i,K}].

    Args:
        resource_level: () scalar
        agent_signals: (K,) noisy signals for this agent

    Returns:
        (K+1,) state vector
    """
    return jnp.concatenate([jnp.array([resource_level]), agent_signals])


# --- Proposal Generation Transform -------------------------------------------

def proposal_generation_transform(state: GraphState) -> GraphState:
    """Generate K proposals from U(0.80, 1.25) and noisy signals per agent.

    Each proposal u_k is a multiplier on the resource level.
    Each agent i sees u_hat_{i,k} = u_k + N(0, sigma_i^2).
    """
    state, key = _split_key(state)
    K = state.global_attrs["K"]
    n_agents = state.node_types.shape[0]
    signal_quality = state.node_attrs["signal_quality"]  # (N,) sigma values

    k1, k2 = jr.split(key)
    # Draw true proposal utilities
    proposals = jr.uniform(k1, (K,), minval=0.80, maxval=1.25)

    # Generate noisy signals: (N, K)
    noise = jr.normal(k2, (n_agents, K)) * signal_quality[:, None]
    signals = proposals[None, :] + noise

    new_global = dict(state.global_attrs)
    new_global["proposals"] = proposals
    new_global["signals"] = signals
    return state.replace(global_attrs=new_global)


# --- Voting Transform --------------------------------------------------------

def make_voting_transform(mechanism: str) -> Transform:
    """Create voting transform for the given mechanism.

    All agents select an action via epsilon-greedy Q-learning.
    For PDD/PRD: actions are proposal indices {0, ..., K-1}.
    For PLD: actions are {0, ..., K-1} (vote for proposal) or K (delegate).
    """
    def voting_transform(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        n_agents = state.node_types.shape[0]
        K = state.global_attrs["K"]
        resource_level = state.global_attrs["resource_level"]
        signals = state.global_attrs["signals"]  # (N, K)
        epsilon = _get_epsilon(state)

        q_weights = state.node_attrs["q_weights"]  # (N, n_actions, state_dim)
        q_bias = state.node_attrs["q_bias"]         # (N, n_actions)

        # Number of valid actions depends on mechanism
        n_valid = K + 1 if mechanism == "pld" else K

        # Build state vectors for all agents: (N, state_dim)
        state_vecs = jax.vmap(
            lambda sigs: _build_state_vec(resource_level, sigs)
        )(signals)

        # Epsilon-greedy action selection, vmapped over agents
        keys = jr.split(key, n_agents)
        actions = jax.vmap(
            lambda w, b, s, k: q_select_action(w, b, s, k, epsilon, n_valid)
        )(q_weights, q_bias, state_vecs, keys)

        # For PLD: agents choosing action K are delegating
        # Resolve delegation: find target, use target's voted proposal
        if mechanism == "pld":
            trust_scores = state.node_attrs["trust_scores"]  # (N, N)
            is_adversarial = state.node_types

            # Find delegation targets for all agents
            targets = jax.vmap(delegate_target)(trust_scores, is_adversarial)

            # An agent delegates if action == K
            is_delegating = (actions == K)

            # Resolve one level of delegation: take target's action
            # If target is also delegating, use target's target (simple 1-hop)
            target_actions = actions[targets]
            target_targets = targets[targets]
            target_of_target_actions = actions[target_targets]

            # Resolve: if my target delegated too, follow one more hop
            resolved_actions = jnp.where(
                target_actions == K,
                target_of_target_actions,
                target_actions,
            )
            # If still delegating after 2 hops, default to action 0
            resolved_actions = jnp.where(resolved_actions == K, 0, resolved_actions)

            # Replace delegating agents' actions with resolved actions
            actions = jnp.where(is_delegating, resolved_actions, actions)
            # Clip to valid proposal range after delegation resolution
            actions = jnp.clip(actions, 0, K - 1)

        state = state.update_node_attrs("last_action", actions)
        return state

    return voting_transform


# --- Aggregation Transforms --------------------------------------------------

def make_aggregation_transform(mechanism: str) -> Transform:
    """Create aggregation transform for the given mechanism.

    PDD: equal-weight plurality voting — most-voted proposal wins.
    PRD: representatives-only plurality — only reps' votes count.
    PLD: trust-weighted plurality — votes weighted by accumulated trust.
    """
    def aggregation_transform(state: GraphState) -> GraphState:
        K = state.global_attrs["K"]
        actions = state.node_attrs["last_action"]  # (N,) proposal indices

        # Count votes per proposal using one-hot encoding
        votes_onehot = jax.nn.one_hot(actions, K)  # (N, K)

        if mechanism == "pdd":
            # Equal-weight plurality
            vote_counts = jnp.sum(votes_onehot, axis=0)  # (K,)

        elif mechanism == "prd":
            # Only representatives vote
            rep_mask = state.node_attrs["rep_mask"]  # (N,)
            vote_counts = jnp.sum(votes_onehot * rep_mask[:, None], axis=0)

        elif mechanism == "pld":
            # Trust-weighted plurality
            trust_scores = state.node_attrs["trust_scores"]  # (N, N)
            # Each agent's weight = sum of trust others have in them
            agent_weights = jnp.sum(trust_scores, axis=0)  # (N,)
            # Normalize
            agent_weights = agent_weights / (jnp.sum(agent_weights) + 1e-8)
            vote_counts = jnp.sum(votes_onehot * agent_weights[:, None], axis=0)

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        # Winner = argmax of vote counts (ties broken by lowest index)
        selected = jnp.argmax(vote_counts)

        new_global = dict(state.global_attrs)
        new_global["selected_proposal"] = selected
        return state.replace(global_attrs=new_global)

    return aggregation_transform


# --- Resource Update Transform ------------------------------------------------

def resource_update_transform(state: GraphState) -> GraphState:
    """Multiplicative resource dynamics: R(t+1) = R(t) * u_{k*}.

    The selected proposal's true utility multiplies the resource level.
    Collapse occurs if R < collapse_threshold (checked in environment).
    """
    proposals = state.global_attrs["proposals"]  # (K,)
    selected = state.global_attrs["selected_proposal"]
    resource = state.global_attrs["resource_level"]

    multiplier = proposals[selected]
    new_resource = resource * multiplier

    new_global = dict(state.global_attrs)
    new_global["resource_level"] = new_resource
    return state.replace(global_attrs=new_global)


# --- Reward Transform ---------------------------------------------------------

def reward_transform(state: GraphState) -> GraphState:
    """Compute rewards based on resource change.

    Aligned agents:     r = R(t+1) - R(t)
    Adversarial agents: r = -(R(t+1) - R(t))

    Dispatched via jnp.where on node_types.
    """
    resource = state.global_attrs["resource_level"]
    proposals = state.global_attrs["proposals"]
    selected = state.global_attrs["selected_proposal"]

    # Resource change from this round's selection
    multiplier = proposals[selected]
    # R_before = R / multiplier (since resource_update already applied)
    r_before = resource / (multiplier + 1e-8)
    delta_r = resource - r_before

    is_adversarial = state.node_types  # 0=coop, 1=adv
    rewards = jnp.where(is_adversarial, -delta_r, delta_r)
    # Broadcast to all agents (same reward signal — it's a collective outcome)
    rewards = jnp.broadcast_to(rewards, (state.node_types.shape[0],))

    state = state.update_node_attrs("last_reward", rewards)
    return state


# --- Q-Learning Transform ----------------------------------------------------

def make_q_learning_transform(mechanism: str) -> Transform:
    """TD(0) update on Q-weights for all agents."""
    def q_learning_transform(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        K = state.global_attrs["K"]
        alpha = state.global_attrs["alpha"]
        gamma = state.global_attrs["gamma"]
        resource_level = state.global_attrs["resource_level"]
        signals = state.global_attrs["signals"]  # (N, K)

        q_weights = state.node_attrs["q_weights"]
        q_bias = state.node_attrs["q_bias"]
        actions = state.node_attrs["last_action"]
        rewards = state.node_attrs["last_reward"]

        n_valid = K + 1 if mechanism == "pld" else K

        # Current state vectors (post-update resource, current signals)
        # Using current signals as next_state approximation
        # (next round will have new proposals, but resource is known)
        state_vecs = jax.vmap(
            lambda sigs: _build_state_vec(resource_level, sigs)
        )(signals)

        # TD(0) update vmapped over agents
        new_weights, new_bias = jax.vmap(
            lambda w, b, s, a, r: q_update(w, b, s, a, r, s, alpha, gamma, n_valid)
        )(q_weights, q_bias, state_vecs, actions, rewards)

        state = state.update_node_attrs("q_weights", new_weights)
        state = state.update_node_attrs("q_bias", new_bias)
        return state

    return q_learning_transform


# --- Trust Update Transform ---------------------------------------------------

def trust_update_transform(state: GraphState) -> GraphState:
    """EMA performance tracking for trust scores.

    Each agent updates trust in every other agent based on what proposal
    that other agent voted for and its true utility.
    """
    proposals = state.global_attrs["proposals"]  # (K,)
    trust_lambda = state.global_attrs["trust_lambda"]
    actions = state.node_attrs["last_action"]  # (N,) proposal indices
    old_trust = state.node_attrs["trust_scores"]  # (N, N)

    # True utility of each agent's voted proposal
    voted_utilities = proposals[actions]  # (N,)
    mean_utility = jnp.mean(proposals)

    # Update each agent's trust in all others via vmap
    new_trust = jax.vmap(
        lambda row: trust_update(row, voted_utilities, mean_utility, trust_lambda)
    )(old_trust)

    state = state.update_node_attrs("trust_scores", new_trust)
    return state


# --- Election Transform (PRD only) -------------------------------------------

def make_election_transform(mechanism: str) -> Transform:
    """PRD: elect R representatives every E rounds by trust-based voting.

    Agents vote for their most-trusted non-self agent as representative.
    Top n_reps agents by vote count become the new representatives.
    """
    def election_transform(state: GraphState) -> GraphState:
        if mechanism != "prd":
            return state

        step = state.global_attrs["step"]
        E = state.global_attrs["election_period"]
        n_reps = state.global_attrs["n_reps"]
        n_agents = state.node_types.shape[0]
        trust_scores = state.node_attrs["trust_scores"]  # (N, N)

        # Zero out self-trust for voting purposes
        mask = 1.0 - jnp.eye(n_agents)
        masked_trust = trust_scores * mask

        # Each agent votes for their most-trusted agent
        votes = jnp.argmax(masked_trust, axis=1)  # (N,)

        # Count votes per agent
        vote_counts = jnp.zeros(n_agents)
        vote_counts = jax.vmap(
            lambda vc, v: vc.at[v].add(1.0),
            in_axes=(None, 0),
            out_axes=0,
        )(vote_counts, votes)
        # Sum across agents
        vote_counts = jnp.sum(jax.nn.one_hot(votes, n_agents), axis=0)

        # Top n_reps become representatives
        _, top_indices = jax.lax.top_k(vote_counts, n_reps)
        new_mask = jnp.zeros(n_agents, dtype=jnp.float32)
        new_mask = new_mask.at[top_indices].set(1.0)

        # Only update on election rounds (step % E == 0)
        is_election = (step % E == 0) & (step > 0)
        old_mask = state.node_attrs["rep_mask"]
        final_mask = jnp.where(is_election, new_mask, old_mask)

        state = state.update_node_attrs("rep_mask", final_mask)
        return state

    return election_transform


# --- Step Counter -------------------------------------------------------------

def step_counter_transform(state: GraphState) -> GraphState:
    """Increment the step counter."""
    new_global = dict(state.global_attrs)
    new_global["step"] = state.global_attrs["step"] + 1
    return state.replace(global_attrs=new_global)


# --- Composition --------------------------------------------------------------

def make_step_transform(mechanism: str = "pdd") -> Transform:
    """Compose the full step pipeline for a given mechanism.

    Pipeline:
        proposal_gen -> voting -> aggregation -> resource_update ->
        reward -> q_learning -> trust_update -> election -> step_counter
    """
    return sequential(
        proposal_generation_transform,
        make_voting_transform(mechanism),
        make_aggregation_transform(mechanism),
        resource_update_transform,
        reward_transform,
        make_q_learning_transform(mechanism),
        trust_update_transform,
        make_election_transform(mechanism),
        step_counter_transform,
    )
