# transformations/top_down/democratic_transforms/election.py
import jax
import jax.numpy as jnp
# import jax.random as jr # Not needed if votes are pre-supplied
from typing import Dict, Any

from algebra.graph import GraphState
from algebra.category import Transform

def create_election_transform(
    candidate_flag_attr: str = "is_delegate", # Node attribute marking potential candidates
    election_votes_attr: str = "agent_votes_for_candidates" # Node attribute storing cast votes for candidates
                                                            # Shape: (num_voters, num_candidates_in_list)
                                                            # The calling code needs to ensure this is populated.
) -> Transform:
    """
    Pure election transform for PRD.
    - Checks election timing.
    - Identifies candidates based on `candidate_flag_attr`.
    - Tallies votes from `election_votes_attr`.
    - Selects top N candidates.
    - Updates representative status and terms.
    
    This transform assumes that agent_votes_for_candidates has been populated
    by a preceding transform (e.g., an LLM decision step where agents vote
    for election candidates).
    """
    def transform(state: GraphState) -> GraphState:
        round_num = state.global_attrs.get("round_num", 0)
        term_length = state.global_attrs.get("prd_election_term_length", 4) # Get term length from globals

        # 1. Check if it's election time
        if state.global_attrs.get("rounds_until_next_election_prd", 0) > 0:
            new_global_attrs = state.global_attrs.copy()
            new_global_attrs["rounds_until_next_election_prd"] -= 1
            
            new_node_attrs = state.node_attrs.copy()
            if "representative_term_remaining" in new_node_attrs:
                new_node_attrs["representative_term_remaining"] = jnp.maximum(
                    0, state.node_attrs["representative_term_remaining"] - 1
                )
            else:
                new_node_attrs["representative_term_remaining"] = jnp.zeros(state.num_nodes, dtype=jnp.int32)
            return state.replace(global_attrs=new_global_attrs, node_attrs=new_node_attrs)

        print(f"[PRD Election] Round {round_num}: Holding new election (Term Length: {term_length}).")
        
        num_agents = state.num_nodes
        if num_agents == 0:
            print(f"[PRD Election] R{round_num}: No agents. Skipping election.")
            new_global_attrs_no_agents = state.global_attrs.copy()
            new_global_attrs_no_agents["rounds_until_next_election_prd"] = term_length -1 if term_length > 0 else 0
            return state.replace(global_attrs=new_global_attrs_no_agents)

        # 2. Identify Candidates
        candidate_mask = state.node_attrs.get(candidate_flag_attr, jnp.zeros(num_agents, dtype=jnp.bool_))
        candidate_original_indices = jnp.where(candidate_mask)[0] # Actual agent IDs of candidates
        num_actual_candidates = candidate_original_indices.shape[0]

        if num_actual_candidates == 0:
            print(f"[PRD Election] R{round_num}: No candidates flagged with '{candidate_flag_attr}'. Skipping election.")
            new_global_attrs_no_cands = state.global_attrs.copy()
            new_global_attrs_no_cands["rounds_until_next_election_prd"] = term_length -1 if term_length > 0 else 0
            new_node_attrs_no_cands = state.node_attrs.copy()
            new_node_attrs_no_cands["is_elected_representative"] = jnp.zeros(num_agents, dtype=jnp.bool_)
            # Ensure terms correctly handled for outgoing reps
            current_terms = new_node_attrs_no_cands.get("representative_term_remaining", jnp.zeros(num_agents, dtype=jnp.int32))
            new_node_attrs_no_cands["representative_term_remaining"] = jnp.maximum(0, current_terms - 1)
            return state.replace(global_attrs=new_global_attrs_no_cands, node_attrs=new_node_attrs_no_cands)

        # 3. Determine Number to Elect
        num_to_elect_config = state.global_attrs.get("prd_num_representatives_to_elect")
        default_num_to_elect = state.global_attrs.get("num_delegates", min(num_actual_candidates, 4))
        num_to_elect = min(num_to_elect_config if isinstance(num_to_elect_config, int) and num_to_elect_config > 0 else default_num_to_elect, num_actual_candidates)

        if num_to_elect == 0 and num_actual_candidates > 0:
             print(f"[PRD Election WARNING] R{round_num}: num_to_elect is 0, but there are candidates. No one will be elected.")

        # 4. Get Cast Votes for Candidates
        # Assumes `agent_votes_for_candidates` is shape (num_agents, num_actual_candidates)
        # where the columns in the vote array correspond to the order of candidates in `candidate_original_indices`.
        agent_votes_for_candidate_list = state.node_attrs.get(election_votes_attr)
        
        if agent_votes_for_candidate_list is None:
            print(f"[PRD Election ERROR] R{round_num}: Missing '{election_votes_attr}' in node_attrs. Cannot tally votes. Defaulting to no elected.")
            total_votes_for_each_candidate_in_list = jnp.zeros(num_actual_candidates, dtype=jnp.int32)
        elif agent_votes_for_candidate_list.shape != (num_agents, num_actual_candidates):
            print(f"[PRD Election ERROR] R{round_num}: Shape of '{election_votes_attr}' is {agent_votes_for_candidate_list.shape}, expected ({num_agents}, {num_actual_candidates}). Defaulting to no elected.")
            total_votes_for_each_candidate_in_list = jnp.zeros(num_actual_candidates, dtype=jnp.int32)
        else:
            # Assuming approval votes (0 or 1). If scores, sum directly.
            total_votes_for_each_candidate_in_list = jnp.sum(agent_votes_for_candidate_list, axis=0) 

        # 5. Select Winners
        elected_agent_ids = jnp.array([], dtype=jnp.int32)
        if num_actual_candidates > 0 and num_to_elect > 0 and total_votes_for_each_candidate_in_list.size > 0 :
            tie_breaker = (-candidate_original_indices.astype(jnp.float32) / (float(num_agents) + 1.0)) * 1e-6 
            scores_for_sorting = total_votes_for_each_candidate_in_list.astype(jnp.float32) + tie_breaker
            sorted_indices_in_candidate_list = jnp.argsort(scores_for_sorting)[::-1]
            elected_indices_in_cand_list = sorted_indices_in_candidate_list[:num_to_elect]
            elected_agent_ids = candidate_original_indices[elected_indices_in_cand_list]
        
        # 6. Update State
        new_node_attrs = state.node_attrs.copy()
        new_node_attrs["is_elected_representative"] = jnp.zeros(num_agents, dtype=jnp.bool_)
        if elected_agent_ids.size > 0:
            new_node_attrs["is_elected_representative"] = new_node_attrs["is_elected_representative"].at[elected_agent_ids].set(True)
        
        current_terms_update = new_node_attrs.get("representative_term_remaining", jnp.zeros(num_agents, dtype=jnp.int32))
        new_node_attrs["representative_term_remaining"] = jnp.where(
            new_node_attrs["is_elected_representative"],
            term_length,
            jnp.maximum(0, current_terms_update - 1) 
        )
        
        new_global_attrs = state.global_attrs.copy()
        new_global_attrs["rounds_until_next_election_prd"] = term_length - 1 if term_length > 0 else 0

        print(f"[PRD Election] R{round_num}: Elected Agent IDs: {elected_agent_ids.tolist()} for a term of {term_length} rounds.")
        # ... (logging adversarial elected count) ...

        return state.replace(node_attrs=new_node_attrs, global_attrs=new_global_attrs)
    return transform