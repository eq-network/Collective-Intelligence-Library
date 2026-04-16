"""
Transform pipeline for the basin stability experiment.

Each transform is GraphState -> GraphState, composed via core.category.sequential().
The mechanism (PDD/PRD/PLD) is selected at composition time, not runtime.

Full step pipeline:
    proposal_gen -> voting -> aggregation -> resource_update ->
    reward -> trust_update -> election -> step_counter -> [metrics]

Agent model: heuristic signal-threshold voting with adaptive trust.
Portfolio yields: Beta-distributed with 4 fixed risk-reward profiles.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState
from core.category import Transform, sequential

from .policies import (
    approval_vote,
    should_delegate,
    delegate_target,
    election_vote,
    trust_update,
)


def _split_key(state: GraphState):
    """Consume and advance the RNG key in global_attrs."""
    key = state.global_attrs["rng_key"]
    key, subkey = jr.split(key)
    new_global = dict(state.global_attrs)
    new_global["rng_key"] = key
    return state.replace(global_attrs=new_global), subkey


# --- Proposal Generation Transform -------------------------------------------

def proposal_generation_transform(state: GraphState) -> GraphState:
    """Generate K portfolio yields from Beta distributions and noisy signals.

    Each portfolio k has yield y_k ~ low_k + (high_k - low_k) * Beta(alpha_k, beta_k).
    Each agent i observes y_hat_{i,k} = y_k + N(0, sigma_i^2).
    """
    state, key = _split_key(state)
    K = state.global_attrs["K"]
    n_agents = state.node_types.shape[0]
    signal_quality = state.node_attrs["signal_quality"]  # (N,) sigma values

    alphas = state.global_attrs["portfolio_alphas"]  # (K,)
    betas = state.global_attrs["portfolio_betas"]     # (K,)
    lows = state.global_attrs["portfolio_lows"]       # (K,)
    highs = state.global_attrs["portfolio_highs"]     # (K,)

    k1, k2 = jr.split(key)
    # Sample from Beta and scale to yield range
    raw = jr.beta(k1, alphas, betas)  # (K,) in [0, 1]
    proposals = lows + (highs - lows) * raw  # (K,) scaled yields

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

    All agents compute approval votes via signal-threshold heuristic.
    For PLD: low-confidence cooperative agents delegate to highest-trust agent.
    Adversarial agents always vote directly (never delegate).
    """
    def voting_transform(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        n_agents = state.node_types.shape[0]
        signals = state.global_attrs["signals"]         # (N, K)
        is_adversarial = state.node_types                # (N,) 0=coop, 1=adv

        # Compute approval votes for all agents via vmap
        approvals, top_choices, confidences = jax.vmap(approval_vote)(
            signals, is_adversarial
        )
        # approvals: (N, K), top_choices: (N,), confidences: (N,)

        # Default: everyone votes directly with unit weight
        vote_weights = jnp.ones(n_agents)

        if mechanism == "pld":
            # SNR-based delegation for cooperative agents only
            signal_quality = state.node_attrs["signal_quality"]  # (N,)
            snr_threshold = state.global_attrs["snr_threshold"]
            trust_scores = state.node_attrs["trust_scores"]      # (N, N)

            # Determine who delegates
            wants_to_delegate = jax.vmap(should_delegate)(
                confidences, signal_quality, jnp.full(n_agents, snr_threshold)
            )
            # Adversarial agents never delegate
            is_delegating = wants_to_delegate & (is_adversarial == 0)

            # Find delegation targets (mask self-trust)
            mask = 1.0 - jnp.eye(n_agents)
            masked_trust = trust_scores * mask
            targets = jax.vmap(delegate_target)(masked_trust, is_adversarial)

            # Delegating agents adopt their target's approvals and top choice
            target_approvals = approvals[targets]
            target_top_choices = top_choices[targets]
            approvals = jnp.where(is_delegating[:, None], target_approvals, approvals)
            top_choices = jnp.where(is_delegating, target_top_choices, top_choices)

            # Compute effective vote weights (paper Eq. 4):
            #   w_j = is_voting_j * (1 + delegation_count_j)
            # Non-transitive: if target also delegates, weight is lost.
            delegation_counts = jnp.zeros(n_agents).at[targets].add(
                is_delegating.astype(jnp.float32)
            )
            is_voting = ~is_delegating
            vote_weights = is_voting.astype(jnp.float32) * (1.0 + delegation_counts)

        state = state.update_node_attrs("approval_votes", approvals)
        state = state.update_node_attrs("last_action", top_choices)
        state = state.update_node_attrs("vote_weight", vote_weights)
        return state

    return voting_transform


# --- Aggregation Transforms --------------------------------------------------

# --- Weight Assignment Transforms (mechanism-specific) ------------------------

def equal_weight_transform(state: GraphState) -> GraphState:
    """PDD: assign equal weight to all agents."""
    n_agents = state.node_types.shape[0]
    return state.update_node_attrs("vote_weight", jnp.ones(n_agents))


def rep_weight_transform(state: GraphState) -> GraphState:
    """PRD: only elected representatives have voting weight."""
    return state.update_node_attrs("vote_weight", state.node_attrs["rep_mask"])


# PLD: vote_weight is already set in the voting transform (delegation logic)


# --- Aggregation Transform (universal, shared across all mechanisms) ----------

def aggregation_transform(state: GraphState) -> GraphState:
    """Weighted plurality vote on each agent's top choice.

    Reads vote_weight from node_attrs — this is the ONLY input that
    varies between mechanisms. The aggregation rule itself is identical.
    """
    K = state.global_attrs["K"]
    actions = state.node_attrs["last_action"]     # (N,) top choice per agent
    weights = state.node_attrs["vote_weight"]     # (N,) mechanism-assigned weight

    votes_onehot = jax.nn.one_hot(actions, K)     # (N, K)
    vote_counts = jnp.sum(votes_onehot * weights[:, None], axis=0)  # (K,)

    selected = jnp.argmax(vote_counts)

    new_global = dict(state.global_attrs)
    new_global["selected_proposal"] = selected
    return state.replace(global_attrs=new_global)


# --- Resource Update Transform ------------------------------------------------

def resource_update_transform(state: GraphState) -> GraphState:
    """Multiplicative resource dynamics: R(t+1) = R(t) * y_{k*}.

    The selected portfolio's realized yield multiplies the resource level.
    Gated by alive flag — if collapsed, resource freezes at last value.
    """
    alive = state.global_attrs["alive"]
    proposals = state.global_attrs["proposals"]  # (K,)
    selected = state.global_attrs["selected_proposal"]
    resource = state.global_attrs["resource_level"]

    multiplier = proposals[selected]
    new_resource = resource * multiplier

    # Check collapse
    threshold = state.global_attrs["collapse_threshold"]
    new_alive = alive * (new_resource >= threshold).astype(jnp.float32)

    # If dead, freeze resource at last value
    new_resource = jnp.where(new_alive > 0.5, new_resource, resource)

    new_global = dict(state.global_attrs)
    new_global["resource_level"] = new_resource
    new_global["alive"] = new_alive
    return state.replace(global_attrs=new_global)


# --- Reward Transform (metrics only) -----------------------------------------

def reward_transform(state: GraphState) -> GraphState:
    """Compute reward from resource change for metrics compatibility.

    Aligned agents:     r = R(t+1) - R(t)
    Adversarial agents: r = -(R(t+1) - R(t))

    No learning update — this just stores the value for metric functions.
    """
    resource = state.global_attrs["resource_level"]
    proposals = state.global_attrs["proposals"]
    selected = state.global_attrs["selected_proposal"]

    # Resource change from this round's selection
    multiplier = proposals[selected]
    r_before = resource / (multiplier + 1e-8)
    delta_r = resource - r_before

    is_adversarial = state.node_types
    n_agents = state.node_types.shape[0]
    rewards = jnp.where(is_adversarial, -delta_r, delta_r)
    rewards = jnp.broadcast_to(rewards, (n_agents,))

    state = state.update_node_attrs("last_reward", rewards)
    return state


# --- Trust Update Transform ---------------------------------------------------

def trust_update_transform(state: GraphState) -> GraphState:
    """EMA performance tracking for trust scores.

    Each agent updates trust in every other agent based on what portfolio
    that other agent voted for and its true utility.

    tracking_lambda controls memory:
        0.9 = predictive (long-term track record)
        0.1 = non-predictive (recency bias / populism)
    """
    proposals = state.global_attrs["proposals"]           # (K,)
    tracking_lambda = state.global_attrs["tracking_lambda"]
    actions = state.node_attrs["last_action"]             # (N,) top choices
    old_trust = state.node_attrs["trust_scores"]          # (N, N)

    # True utility of each agent's preferred portfolio
    voted_utilities = proposals[actions]  # (N,)
    mean_utility = jnp.mean(proposals)

    # Update each agent's trust in all others via vmap
    new_trust = jax.vmap(
        lambda row: trust_update(row, voted_utilities, mean_utility, tracking_lambda)
    )(old_trust)

    state = state.update_node_attrs("trust_scores", new_trust)
    return state


# --- Election Transform (PRD only) -------------------------------------------

def make_election_transform(mechanism: str) -> Transform:
    """PRD: elect R representatives every E rounds by trust-based voting.

    Cooperative agents vote for their most-trusted non-self agent.
    Adversarial agents vote for their least-trusted (to install bad reps).
    Top n_reps agents by vote count become new representatives.
    """
    def election_transform(state: GraphState) -> GraphState:
        if mechanism != "prd":
            return state

        step = state.global_attrs["step"]
        E = state.global_attrs["election_period"]
        n_reps = state.global_attrs["n_reps"]
        n_agents = state.node_types.shape[0]
        trust_scores = state.node_attrs["trust_scores"]  # (N, N)
        is_adversarial = state.node_types                 # (N,)

        # Mask self-trust to zero so agents can't vote for themselves
        mask = 1.0 - jnp.eye(n_agents)
        masked_trust = trust_scores * mask

        # Optimal bloc voting: each faction targets ceil(n_reps/2) seats
        target_seats = (n_reps + 1) // 2  # ceil(n_reps / 2)
        agent_indices = jnp.arange(n_agents)
        target_seats_arr = jnp.full(n_agents, target_seats)

        # Type-aware bloc election voting (spreads votes over target seats)
        votes = jax.vmap(election_vote)(
            masked_trust, is_adversarial, agent_indices, target_seats_arr
        )  # (N,)

        # Count votes per agent via one-hot
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

def make_step_transform(mechanism: str = "pdd", metrics: dict = None) -> Transform:
    """Compose the full step pipeline for a given mechanism.

    Shared transforms (identical across all mechanisms):
        proposal_gen, voting, aggregation, resource_update,
        reward, trust_update, step_counter

    Mechanism-specific transforms (composed in, the ONLY difference):
        PDD: equal_weight
        PRD: rep_weight + election
        PLD: delegation (inside voting transform) — sets vote_weight

    Pipeline:
        PDD: proposal_gen >> voting >> equal_weight >> aggregation >> resource >> reward >> trust >> step
        PRD: proposal_gen >> voting >> rep_weight >> aggregation >> resource >> reward >> trust >> election >> step
        PLD: proposal_gen >> voting(+delegation) >> aggregation >> resource >> reward >> trust >> step
    """
    # Phase 1: generate proposals and signals, all agents vote
    transforms = [
        proposal_generation_transform,
        make_voting_transform(mechanism),
    ]

    # Phase 2: assign weights (mechanism-specific)
    if mechanism == "pdd":
        transforms.append(equal_weight_transform)
    elif mechanism == "prd":
        transforms.append(rep_weight_transform)
    # PLD: vote_weight already set by voting transform (delegation logic)

    # Phase 3: shared aggregation + dynamics (identical for all mechanisms)
    transforms.extend([
        aggregation_transform,
        resource_update_transform,
        reward_transform,
        trust_update_transform,
    ])

    # Phase 4: election for PRD (updates rep_mask for next round)
    if mechanism == "prd":
        transforms.append(make_election_transform(mechanism))

    # Phase 5: metrics + step counter
    if metrics:
        from metrics.transform import make_metrics_transform
        transforms.append(make_metrics_transform(metrics))

    transforms.append(step_counter_transform)

    return sequential(*transforms)
