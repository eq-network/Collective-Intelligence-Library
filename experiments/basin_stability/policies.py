"""
Pure policy functions for the basin stability experiment.

All functions operate on single-agent data and are designed for jax.vmap.
Agent-type dispatch (cooperative vs adversarial) uses jnp.where in transforms.

Agent model: Linear Q-learning with TD(0) updates.
    Q(s, a) = w_a^T s + b_a
    s = [R(t), u_hat_1, ..., u_hat_K]
"""
import jax
import jax.numpy as jnp
import jax.random as jr


def q_values(q_weights, q_bias, state_vec):
    """Compute Q-values for all actions given state vector.

    Args:
        q_weights: (n_actions, state_dim) weight matrix
        q_bias: (n_actions,) bias vector
        state_vec: (state_dim,) state vector [R, u_hat_1, ..., u_hat_K]

    Returns:
        (n_actions,) Q-values
    """
    return q_weights @ state_vec + q_bias


def q_select_action(q_weights, q_bias, state_vec, key, epsilon, n_valid_actions):
    """Epsilon-greedy action selection from linear Q-function.

    Args:
        q_weights: (n_actions, state_dim)
        q_bias: (n_actions,)
        state_vec: (state_dim,)
        key: PRNG key
        epsilon: exploration rate
        n_valid_actions: number of valid actions (K for PDD/PRD, K+1 for PLD)

    Returns:
        action: () integer action index
    """
    qvals = q_values(q_weights, q_bias, state_vec)
    # Mask invalid actions (set to -inf so they're never selected)
    mask = jnp.arange(qvals.shape[0]) < n_valid_actions
    qvals = jnp.where(mask, qvals, -jnp.inf)

    k1, k2 = jr.split(key)
    greedy_action = jnp.argmax(qvals)
    random_action = jr.randint(k1, (), 0, n_valid_actions)
    explore = jr.uniform(k2) < epsilon
    return jnp.where(explore, random_action, greedy_action)


def q_update(q_weights, q_bias, state_vec, action, reward, next_state_vec,
             alpha, gamma, n_valid_actions):
    """TD(0) update on Q-weights for the chosen action.

    Updates only the weights/bias for the selected action:
        delta = reward + gamma * max_a' Q(s', a') - Q(s, a)
        w_a += alpha * delta * s
        b_a += alpha * delta

    Args:
        q_weights: (n_actions, state_dim)
        q_bias: (n_actions,)
        state_vec: (state_dim,) current state
        action: () integer action taken
        reward: () scalar reward received
        next_state_vec: (state_dim,) next state
        alpha: learning rate
        gamma: discount factor
        n_valid_actions: number of valid actions

    Returns:
        new_q_weights: (n_actions, state_dim)
        new_q_bias: (n_actions,)
    """
    # Current Q-value for chosen action
    current_q = q_weights[action] @ state_vec + q_bias[action]

    # Max Q-value in next state (over valid actions only)
    next_qvals = q_weights @ next_state_vec + q_bias
    mask = jnp.arange(next_qvals.shape[0]) < n_valid_actions
    next_qvals = jnp.where(mask, next_qvals, -jnp.inf)
    max_next_q = jnp.max(next_qvals)

    # TD error
    delta = reward + gamma * max_next_q - current_q

    # Update only the row for the chosen action
    new_weights = q_weights.at[action].add(alpha * delta * state_vec)
    new_bias = q_bias.at[action].add(alpha * delta)

    return new_weights, new_bias


def trust_update(trust_scores, voted_proposal_utility, mean_utility, trust_lambda):
    """Update trust scores for a single agent based on observed performance.

    Trust of agent i in agent j is updated via EMA:
        rho_j = u_{a_j} / mean(u_k)  (performance ratio)
        trust_{i,j} = lambda * trust_{i,j} + (1 - lambda) * rho_j

    Args:
        trust_scores: (N,) this agent's trust in all other agents
        voted_proposal_utility: (N,) true utility of each agent's voted proposal
        mean_utility: () mean utility across all proposals
        trust_lambda: EMA decay

    Returns:
        (N,) updated trust scores
    """
    performance = voted_proposal_utility / (mean_utility + 1e-8)
    return trust_lambda * trust_scores + (1.0 - trust_lambda) * performance


def delegate_target(trust_scores_row, is_adversarial):
    """Select delegation target for PLD.

    Cooperative agents delegate to their most-trusted agent.
    Adversarial agents delegate to their least-trusted (to amplify bad choices).

    Args:
        trust_scores_row: (N,) this agent's trust in all agents
        is_adversarial: () boolean

    Returns:
        target_idx: () integer index of delegation target
    """
    return jnp.where(
        is_adversarial,
        jnp.argmin(trust_scores_row),
        jnp.argmax(trust_scores_row),
    )
