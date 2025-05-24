# transformations/top_down/democratic_transforms/election.py
import jax
import jax.numpy as jnp
import jax.random as jr
import re # For parsing LLM responses
from typing import Dict, Any, List

from core.graph import GraphState
from core.category import Transform

# For LLM-based election voting (if desired)
from services.llm import LLMService 
# from environments.democracy.configuration import PromptConfig # If using prompts for election

def _parse_election_votes_from_llm(response: str, num_candidates: int) -> List[int]:
    """
    Parses an LLM string response to extract election votes.
    Expects a response containing a list-like structure, e.g., "Candidate Approvals: [0,1,0,1]".
    Returns a list of 0s or 1s with length equal to num_candidates.
    This parser aims to be robust against common LLM output variations.
    """
    # Attempt to find a list-like structure, e.g., "[0,1,0]" or "Approvals: [1, 0]"
    # The regex (.*?) is non-greedy to capture only the content of the first found brackets.
    match = re.search(r"\[(.*?)\]", response)

    # If no list pattern is found in the response string.
    if not match:
        print(f"Warning: _parse_election_votes_from_llm: Could not find list pattern '[]' in LLM response: '{response}'. Defaulting to no approvals.")
        # Returns [0,0,...] for num_candidates > 0, or [] if num_candidates == 0.
        return [0] * num_candidates

    content = match.group(1).strip() # Get content within brackets and strip whitespace

    # Handle the case where no candidates are expected.
    if num_candidates == 0:
        if content == "": # LLM correctly responded with an empty list for 0 candidates.
            return []
        else: # LLM responded with content, but 0 candidates were expected.
            print(f"Warning: _parse_election_votes_from_llm: LLM response '{response}' has content but expected empty list for 0 candidates. Defaulting to [].")
            return []

    # From this point, num_candidates > 0.
    # The goal is to return a list of 0s and 1s of length num_candidates.
    try:
        parsed_votes: List[int] = []
        if not content: # Handles "[]" when num_candidates > 0 (i.e., LLM returned an empty list of approvals).
            # parsed_votes remains empty. The length check later will ensure it becomes [0]*num_candidates.
            pass
        else:
            # Split the content by comma, strip whitespace from each part.
            str_vote_values = [s.strip() for s in content.split(',')]
            for s_vote in str_vote_values:
                if s_vote == '1':
                    parsed_votes.append(1)
                elif s_vote == '0':
                    parsed_votes.append(0)
                else:
                    # Handle invalid or empty strings (e.g., from "1,,0" or "1, gibberish, 0") as 0.
                    print(f"Warning: _parse_election_votes_from_llm: Invalid or empty vote value '{s_vote}' in '{content}' from response '{response}'. Treating as 0.")
                    parsed_votes.append(0)
        
        # Ensure the final list of votes has the correct length.
        if len(parsed_votes) != num_candidates:
            print(f"Warning: _parse_election_votes_from_llm: Parsed {len(parsed_votes)} votes, but expected {num_candidates} for response '{response}'. Defaulting to list of {num_candidates} zeros.")
            return [0] * num_candidates
        
        return parsed_votes
    except Exception as e:
        # Catch any unexpected error during the parsing process.
        print(f"Error: _parse_election_votes_from_llm: Unexpected error during parsing of '{response}': {e}. Defaulting to list of {num_candidates} zeros.")
        return [0] * num_candidates

def create_election_transform(
    # llm_service: Optional[LLMService] = None, # If LLM involved in election voting
    # prompt_config: Optional[PromptConfig] = None # If LLM involved
    election_logic: str = "random_approval" # or "llm_approval" or "highest_cog_resource"
) -> Transform:
    """
    Simulates an election process for PRD.
    - Identifies candidates (those with is_delegate == True).
    - Agents vote for candidates.
    - Top N candidates become elected representatives.
    - Resets their term length.
    """
    def transform(state: GraphState) -> GraphState:
        # Only run election if it's time
        if state.global_attrs.get("rounds_until_next_election_prd", 0) > 0:
            new_global_attrs = state.global_attrs.copy()
            new_global_attrs["rounds_until_next_election_prd"] -= 1
            # Decrement term for existing reps
            new_node_attrs = state.node_attrs.copy()
            new_node_attrs["representative_term_remaining"] = jnp.maximum(0, state.node_attrs["representative_term_remaining"] - 1)
            return state.replace(global_attrs=new_global_attrs, node_attrs=new_node_attrs)

        print(f"[PRD Election] Round {state.global_attrs.get('round_num', 0)}: Holding new election.")
        
        num_agents = state.num_nodes
        candidate_mask = state.node_attrs["is_delegate"] # Agents who are delegates are candidates
        candidate_indices = jnp.where(candidate_mask)[0]
        num_candidates = len(candidate_indices)

        if num_candidates == 0:
            print("[PRD Election] No candidates available. Skipping election.")
            # Reset election countdown for next round to try again
            new_global_attrs = state.global_attrs.copy()
            new_global_attrs["rounds_until_next_election_prd"] = state.global_attrs["prd_election_term_length"]
            return state.replace(global_attrs=new_global_attrs)

        num_to_elect = state.global_attrs["prd_num_representatives_to_elect"]

        # --- Voting Phase ---
        # Each agent votes on the candidates.
        # For simplicity, let's do random approval voting by each agent for now.
        # A more complex version would involve LLMs or strategic voting.
        
        # candidate_approvals[voter_idx, candidate_idx_in_list]
        all_candidate_votes = jnp.zeros((num_agents, num_candidates), dtype=jnp.int32)
        
        key_base = jr.PRNGKey(state.global_attrs.get("simulation_seed",0) + state.global_attrs.get("round_num",0) + 1000)

        if election_logic == "random_approval":
            # Each agent randomly approves ~50% of candidates
            keys = jr.split(key_base, num_agents)
            for i in range(num_agents):
                # Adversarial agents might vote strategically (e.g., approve bad candidates, or only their own kind)
                is_voter_adversarial = state.node_attrs["is_adversarial"][i]
                if is_voter_adversarial:
                    # Simplistic: adversarial voters approve adversarial candidates if any, else random
                    adv_candidates_present = jnp.any(state.node_attrs["is_adversarial"][candidate_indices])
                    if adv_candidates_present:
                        # Approve only adversarial candidates
                        approvals = state.node_attrs["is_adversarial"][candidate_indices].astype(jnp.int32)
                    else: # No adversarial candidates, vote randomly to disrupt
                        approvals = jr.bernoulli(keys[i], 0.5, shape=(num_candidates,)).astype(jnp.int32)
                else: # Honest voters
                    approvals = jr.bernoulli(keys[i], 0.5, shape=(num_candidates,)).astype(jnp.int32)
                all_candidate_votes = all_candidate_votes.at[i, :].set(approvals)
        
        elif election_logic == "highest_cog_resource": # Non-adversarial agents vote for highest cog resource candidates
             # This is a very simple heuristic for honest agents
            candidate_cog_resources = state.node_attrs["cognitive_resources"][candidate_indices]
            # Honest voters approve top N candidates by cog resources
            # For simplicity, let's say they approve candidates with cog_resources > median cog_resource of candidates
            if num_candidates > 0:
                median_cog_res = jnp.median(candidate_cog_resources)
                honest_approvals = (candidate_cog_resources >= median_cog_res).astype(jnp.int32)
            else:
                honest_approvals = jnp.zeros(num_candidates, dtype=jnp.int32)

            for i in range(num_agents):
                is_voter_adversarial = state.node_attrs["is_adversarial"][i]
                if is_voter_adversarial:
                    # Adversarial still votes randomly or for adversarial candidates
                    adv_candidates_present = jnp.any(state.node_attrs["is_adversarial"][candidate_indices])
                    if adv_candidates_present:
                        approvals = state.node_attrs["is_adversarial"][candidate_indices].astype(jnp.int32)
                    else:
                        approvals = jr.bernoulli(keys[i], 0.5, shape=(num_candidates,)).astype(jnp.int32)
                else: # Honest voters use cog resource heuristic
                    approvals = honest_approvals
                all_candidate_votes = all_candidate_votes.at[i, :].set(approvals)
        # Add LLM based voting here if desired later

        # Tally votes for each candidate
        total_votes_for_candidate = jnp.sum(all_candidate_votes, axis=0) # Sum approvals per candidate

        # Select top N candidates
        # Handle ties by preferring lower original agent ID (deterministic tie-breaking)
        # To do this with JAX, it's a bit tricky. A simpler way for now is to add small random noise or use original index.
        # For stable sorting, we can use a composite sort key: (votes, -original_index)
        # JAX's `argsort` is stable.
        # We want highest votes. If votes are equal, argsort on -candidate_indices means lower original ID wins.
        
        # Tie-breaking: add a small decreasing value based on original index to favor lower original IDs
        # This ensures deterministic tie-breaking. Max num_candidates is num_agents.
        tie_breaker = -candidate_indices.astype(jnp.float32) / (num_agents + 1) 
        sorted_candidate_indices_in_list = jnp.argsort(total_votes_for_candidate + tie_breaker)[::-1] # Descending sort

        elected_candidate_list_indices = sorted_candidate_indices_in_list[:num_to_elect]
        elected_agent_ids = candidate_indices[elected_candidate_list_indices]

        # Update state
        new_node_attrs = state.node_attrs.copy()
        new_node_attrs["is_elected_representative"] = jnp.zeros(num_agents, dtype=jnp.bool_)
        new_node_attrs["is_elected_representative"] = new_node_attrs["is_elected_representative"].at[elected_agent_ids].set(True)
        
        new_node_attrs["representative_term_remaining"] = jnp.where(
            new_node_attrs["is_elected_representative"],
            state.global_attrs["prd_election_term_length"],
            jnp.maximum(0, new_node_attrs["representative_term_remaining"] -1) # Others continue countdown or stay 0
        )
        
        new_global_attrs = state.global_attrs.copy()
        new_global_attrs["rounds_until_next_election_prd"] = state.global_attrs["prd_election_term_length"] -1 # -1 because this round counts

        print(f"[PRD Election] Elected Representatives: {elected_agent_ids.tolist()}")
        adv_elected = new_node_attrs["is_adversarial"][elected_agent_ids].sum()
        print(f"[PRD Election] {adv_elected} adversarial agents elected out of {len(elected_agent_ids)}.")

        return state.replace(node_attrs=new_node_attrs, global_attrs=new_global_attrs)

    return transform