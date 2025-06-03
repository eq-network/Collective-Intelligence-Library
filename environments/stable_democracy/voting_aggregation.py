# environments/stable_democracy/voting_aggregation.py
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, List, Tuple
from core.graph import GraphState
from .configuration import PORTFOLIO_VOTE_TIE_BREAKING_STRATEGY

def _resolve_portfolio_tie(
    aggregated_votes: jnp.ndarray,
    key: jnp.ndarray,
    strategy: str = PORTFOLIO_VOTE_TIE_BREAKING_STRATEGY
) -> int:
    """
    Resolves ties in portfolio voting based on the configured strategy.
    
    Args:
        aggregated_votes: Array of vote counts for each portfolio
        key: JAX random key for random tie-breaking
        strategy: Either "random" or "lowest_index"
        
    Returns:
        Index of the winning portfolio
    """
    max_votes = jnp.max(aggregated_votes)
    tied_indices = jnp.where(aggregated_votes == max_votes)[0]
    
    if len(tied_indices) == 1:
        return int(tied_indices[0])
        
    if strategy == "random":
        # Use JAX random choice for deterministic but random selection
        return int(tied_indices[jr.randint(key, shape=(1,), minval=0, maxval=len(tied_indices))[0]])
    else:  # "lowest_index" or any other value
        return int(jnp.min(tied_indices))

def _portfolio_vote_aggregator_stable(state: GraphState, transform_config: Dict[str, Any]) -> jnp.ndarray:
    """
    Aggregates votes for stable democracy, considering participation status and mechanism.

    - PDD: Sums votes from agents where 'can_participate_this_round' is True.
    - PRD: Sums votes from agents where 'is_elected_representative' is True 
           (implicitly they must also be able to participate if that's a general rule,
            but election status is primary for PRD voting power).
    - PLD: Sums 'voting_power' weighted votes from agents where 'can_participate_this_round' is True
           (or a more nuanced PLD rule for who can cast weighted votes).
    """
    agent_portfolio_votes = state.node_attrs["agent_portfolio_votes"]  # Shape: (num_agents, num_portfolios)
    can_participate = state.node_attrs["can_participate_this_round"] # Shape: (num_agents,)
    num_agents = state.num_nodes
    num_portfolios = agent_portfolio_votes.shape[1]

    if num_agents == 0:
        return jnp.zeros(num_portfolios, dtype=jnp.float32)

    mechanism_type = transform_config.get("mechanism_type", "PDD") # Default to PDD if not specified

    aggregated_votes = jnp.zeros(num_portfolios, dtype=jnp.float32)

    if mechanism_type == "PDD":
        # Only consider votes from participating agents
        participating_votes = agent_portfolio_votes * can_participate[:, jnp.newaxis]
        aggregated_votes = jnp.sum(participating_votes, axis=0)
    elif mechanism_type == "PRD":
        is_elected_representative = state.node_attrs["is_elected_representative"]
        # In PRD, only elected representatives' votes count.
        # We assume elected representatives automatically participate in voting if an election just happened
        # or their term is active. The `can_participate_this_round` might be an additional filter
        # if representatives can also choose to abstain based on the participation model.
        # For simplicity here, let's assume elected reps' votes are primary.
        # A stricter version would be: is_elected_representative & can_participate
        
        # Mask for agents who are elected AND participating
        eligible_prd_voters_mask = is_elected_representative & can_participate

        representative_votes = agent_portfolio_votes * eligible_prd_voters_mask[:, jnp.newaxis]
        aggregated_votes = jnp.sum(representative_votes, axis=0)
    elif mechanism_type == "PLD":
        voting_power = state.node_attrs["voting_power"]
        # Only consider weighted votes from participating agents
        # Non-participating agents might have transferred their power, which is reflected in
        # the `voting_power` of participating delegates.
        weighted_votes = agent_portfolio_votes * voting_power[:, jnp.newaxis]
        participating_weighted_votes = weighted_votes * can_participate[:, jnp.newaxis]
        aggregated_votes = jnp.sum(participating_weighted_votes, axis=0)
    else:
        raise ValueError(f"Unknown mechanism_type for stable vote aggregation: {mechanism_type}")

    # Get the random key from state for tie-breaking
    key = state.global_attrs.get("random_key", jr.PRNGKey(0))
    
    # Find the winning portfolio index using tie-breaking
    winning_index = _resolve_portfolio_tie(aggregated_votes, key)
    
    # Create a one-hot array with the winning portfolio
    result = jnp.zeros_like(aggregated_votes)
    result = result.at[winning_index].set(1.0)
    
    return result.astype(jnp.float32)