# transformations/top_down/democratic_transforms/participation.py
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Callable

from core.graph import GraphState
from core.category import Transform # Assuming Transform is defined in core.category

def create_participation_constraint_transform() -> Transform:
    """
    Determines which agents can participate in the current round based on
    their role, participation rates, and temporal correlation with their
    participation in the previous round.

    Assumes `state.global_attrs` contains 'participation_settings' (ParticipationConfig).
    Reads `state.node_attrs['participated_last_round']`.
    Writes `state.node_attrs['can_participate_this_round']`,
            `state.node_attrs['participated_last_round']` (for next iter),
            and updates `state.node_attrs['participation_history']`.
    """
    def transform(state: GraphState) -> GraphState:
        participation_config = state.global_attrs.get("participation_settings")

        num_agents = state.num_nodes
        if num_agents == 0:
            return state # No agents to process

        round_num = state.global_attrs.get("round_num", 0)
        sim_seed = state.global_attrs.get("simulation_seed", 0)
        
        key_base = jr.PRNGKey(sim_seed + round_num + participation_config.participation_seed_offset)
        keys = jr.split(key_base, num_agents)

        is_delegate = state.node_attrs["is_delegate"]
        participated_last_round = state.node_attrs["participated_last_round"]
        
        can_participate_this_round = jnp.zeros(num_agents, dtype=jnp.bool_)

        for i in range(num_agents):
            base_rate = participation_config.delegate_participation_rate if is_delegate[i] \
                        else participation_config.voter_participation_rate

            # Apply temporal correlation
            # If agent participated last round, increase prob of participating this round.
            # If agent did NOT participate last round, decrease prob of participating this round.
            # The strength of this effect is temporal_correlation_strength.
            # P(participate_t | participated_t-1) = base_rate + (1-base_rate)*strength
            # P(participate_t | not participated_t-1) = base_rate - base_rate*strength
            # These need to be probabilities, so ensure they are in [0,1]
            
            prob_participate = base_rate
            if round_num > 0: # Temporal correlation only applies after the first round
                if participated_last_round[i]:
                    prob_participate = base_rate + (1.0 - base_rate) * participation_config.temporal_correlation_strength
                else:
                    prob_participate = base_rate - base_rate * participation_config.temporal_correlation_strength
            
            prob_participate = jnp.clip(prob_participate, 0.0, 1.0)
            
            can_participate_this_round = can_participate_this_round.at[i].set(
                jr.bernoulli(keys[i], prob_participate)
            )

        # Update node attributes
        new_node_attrs = state.node_attrs.copy()
        new_node_attrs["can_participate_this_round"] = can_participate_this_round
        # The 'participated_last_round' for the *next* round is what we just decided for 'can_participate_this_round'
        new_node_attrs["participated_last_round"] = can_participate_this_round 
        new_node_attrs["participation_history"] = state.node_attrs["participation_history"] + can_participate_this_round.astype(jnp.int32)
        # Reset forced_delegation_this_round at the start of participation decision
        new_node_attrs["forced_delegation_this_round"] = jnp.zeros(num_agents, dtype=jnp.bool_)


        # Edge case handling: what if all delegates are non-participating?
        # This is a complex policy decision. For now, the transform just sets participation.
        # The downstream agent decision logic (especially for PLD) will need to handle it.
        num_participating_delegates = jnp.sum(can_participate_this_round & is_delegate)
        if jnp.sum(is_delegate) > 0 and num_participating_delegates == 0:
            print(f"[Round {round_num}] Warning: All delegates are non-participating this round.")
        
        # Print participation summary
        num_participating_total = jnp.sum(can_participate_this_round)
        print(f"[Round {round_num}] Participation: {num_participating_total}/{num_agents} agents. "
              f"Delegates participating: {num_participating_delegates}/{jnp.sum(is_delegate)}. "
              f"Voters participating: {jnp.sum(can_participate_this_round & (~is_delegate))}/{jnp.sum(~is_delegate)}.")

        return state.replace(node_attrs=new_node_attrs)

    return transform