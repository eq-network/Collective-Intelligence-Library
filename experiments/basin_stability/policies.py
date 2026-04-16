"""
Pure policy functions for the basin stability experiment.

All functions operate on single-agent data and are designed for jax.vmap.
Agent-type dispatch (cooperative vs adversarial) uses jnp.where in transforms.

Agent model: Heuristic signal-threshold voting with adaptive trust dynamics.
    - Cooperative agents approve high-signal portfolios, delegate when uncertain.
    - Adversarial agents approve low-signal portfolios, never delegate.
    - Trust scores update via EMA on observed performance.
"""
import jax.numpy as jnp
import jax.random as jr


def approval_vote(signals, is_adversarial):
    """Signal-threshold approval voting.

    Cooperative agents approve the top half of portfolios (by signal).
    Adversarial agents approve the bottom half.

    Args:
        signals: (K,) noisy signals for each portfolio
        is_adversarial: () boolean (0 or 1)

    Returns:
        approvals: (K,) binary {0, 1} approval vector
        top_choice: () int — index of best (coop) or worst (adv) portfolio
        confidence: () float in [0, 1] — normalized gap between top-2 signals
    """
    K = signals.shape[0]

    # Rank-based threshold: approve portfolios in the top half by signal
    # For K=4: approve the 2 highest-signal portfolios
    sorted_signals = jnp.sort(signals)  # ascending
    threshold = sorted_signals[K // 2]  # median value

    coop_approvals = (signals >= threshold).astype(jnp.float32)
    adv_approvals = (signals < threshold).astype(jnp.float32)
    approvals = jnp.where(is_adversarial, adv_approvals, coop_approvals)

    # Ensure at least 1 portfolio is approved
    any_approved = jnp.sum(approvals) > 0
    coop_fallback = (signals == jnp.max(signals)).astype(jnp.float32)
    adv_fallback = (signals == jnp.min(signals)).astype(jnp.float32)
    fallback = jnp.where(is_adversarial, adv_fallback, coop_fallback)
    approvals = jnp.where(any_approved, approvals, fallback)

    # Top choice: best portfolio for cooperative, worst for adversarial
    top_choice = jnp.where(is_adversarial, jnp.argmin(signals), jnp.argmax(signals))

    # Confidence: normalized gap between top-2 signals (how decisive the ranking is)
    sorted_desc = jnp.sort(signals)[::-1]
    gap = sorted_desc[0] - sorted_desc[1]
    signal_range = sorted_desc[0] - sorted_desc[-1] + 1e-8
    confidence = gap / signal_range

    return approvals, top_choice, confidence


def should_delegate(confidence, sigma_i, snr_threshold):
    """SNR-based delegation decision.

    Delegate when signal confidence is low relative to the agent's noise level.
    High-noise agents (sigma=0.50) delegate often; low-noise (sigma=0.05) rarely.

    Args:
        confidence: () normalized signal gap from approval_vote
        sigma_i: () this agent's signal noise standard deviation
        snr_threshold: () global SNR multiplier (higher = more delegation)

    Returns:
        () boolean — True if agent should delegate
    """
    return confidence < snr_threshold * sigma_i


def delegate_target(trust_scores_row, is_adversarial):
    """Select delegation target for PLD.

    Cooperative agents delegate to their most-trusted agent.
    Adversarial agents delegate to their least-trusted (to amplify bad choices).
    (In practice, adversarial agents never delegate — enforced in transforms.)

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


def election_vote(trust_scores_row, is_adversarial, agent_idx, target_seats):
    """PRD election vote — optimal bloc voting strategy.

    Each faction targets ceil(n_reps/2) seats for majority control.
    Cooperative agents target the most-trusted candidates.
    Adversarial agents target the least-trusted candidates.
    Votes spread evenly across target candidates via agent index modulo.

    This is game-theoretically optimal for committee capture under
    plurality voting (bloc voting strategy from social choice theory).

    Args:
        trust_scores_row: (N,) this agent's trust in all agents (self masked)
        is_adversarial: () boolean
        agent_idx: () this agent's index (for vote spreading)
        target_seats: () number of seats to target (ceil(n_reps/2))

    Returns:
        vote: () int — index of agent voted for
    """
    # Rank candidates: cooperative targets highest trust, adversarial targets lowest
    # argsort ascending: index 0 = least trusted, index -1 = most trusted
    ranked = jnp.argsort(trust_scores_row)

    # Adversarial: pick from bottom target_seats (least trusted)
    # Cooperative: pick from top target_seats (most trusted)
    N = trust_scores_row.shape[0]
    adv_target = ranked[agent_idx % target_seats]              # bottom target_seats
    coop_target = ranked[N - 1 - (agent_idx % target_seats)]   # top target_seats

    return jnp.where(is_adversarial, adv_target, coop_target)


def trust_update(trust_scores, voted_proposal_utility, mean_utility, tracking_lambda):
    """Update trust scores for a single agent based on observed performance.

    Trust of agent i in agent j is updated via EMA:
        rho_j = u_{a_j} / mean(u_k)  (performance ratio)
        trust_{i,j} = lambda * trust_{i,j} + (1 - lambda) * rho_j

    High lambda (0.9) = long memory (predictive tracking).
    Low lambda (0.1) = recency bias (non-predictive / populist).

    Args:
        trust_scores: (N,) this agent's trust in all other agents
        voted_proposal_utility: (N,) true utility of each agent's voted proposal
        mean_utility: () mean utility across all proposals
        tracking_lambda: EMA decay parameter

    Returns:
        (N,) updated trust scores
    """
    performance = voted_proposal_utility / (mean_utility + 1e-8)
    return tracking_lambda * trust_scores + (1.0 - tracking_lambda) * performance
