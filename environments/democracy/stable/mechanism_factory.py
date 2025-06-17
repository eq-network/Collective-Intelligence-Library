# environments/stable_democracy/mechanism_factory.py
from typing import Literal, Dict, Any, Optional, Callable, List, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from algebra.category import Transform, sequential
from algebra.graph import GraphState

from environments.random.mechanism_factory import (
    create_start_of_round_housekeeping_transform,
    _portfolio_resource_calculator, # REUSING this one
    create_agent_performance_update_transform,
)

# Configs specific to stable_democracy
from .configuration import (
    StablePortfolioDemocracyConfig, ParticipationConfig, LockedValueConfig,
    CropConfig, PortfolioStrategyConfig, StablePromptConfig # Use the renamed StablePromptConfig
)

from services.llm import LLMService # Assuming services.llm is accessible

# Transforms - some might be shared, some specific or adapted
from transformations.top_down.democratic_transforms.participation import create_participation_constraint_transform
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform
from transformations.top_down.democratic_transforms.election import create_election_transform

# For initialization and yield fetching within housekeeping
from .initialization import get_stable_true_expected_yields_for_round

# Optimality analysis (if used in stable prompts)
from environments.random.optimality_analysis import (
    OptimalityCalculator,
    generate_optimality_prompt_text
)

from environments.stable.voting_aggregation import _portfolio_vote_aggregator_stable # NEWLY CREATED

# --- Helper Transforms ---
def create_stable_housekeeping_transform() -> Transform:
    def transform(state: GraphState) -> GraphState:
        new_global_attrs = state.global_attrs.copy()
        current_round = new_global_attrs.get("round_num", -1)
        new_global_attrs["round_num"] = current_round + 1
        
        true_expected_yields_for_round = get_stable_true_expected_yields_for_round(
            new_global_attrs["round_num"],
            state.global_attrs["crop_configs"],
            state.global_attrs.get("locked_value_settings")
        )

        new_global_attrs["current_true_expected_crop_yields"] = true_expected_yields_for_round
        # In the stable environment, actual yields are the same as true expected yields for the round.
        new_global_attrs["current_actual_crop_yields"] = true_expected_yields_for_round

        # Reset per-round node attrs if any (e.g. tokens_spent_current_round for compatibility)
        new_node_attrs = state.node_attrs.copy()
        if "tokens_spent_current_round" in new_node_attrs:
             new_node_attrs["tokens_spent_current_round"] = jnp.zeros_like(new_node_attrs["tokens_spent_current_round"])
        
        return state.replace(global_attrs=new_global_attrs, node_attrs=new_node_attrs)
    return transform


# --- Prediction Signal Generation Transform for Stable System ---
def create_stable_prediction_signal_transform() -> Transform:
    """
    Generates agent-specific prediction signals for the stable democracy system.
    In this stable configuration, all agents receive perfect prediction signals,
    equivalent to the current_true_expected_crop_yields.
    """
    def transform(state: GraphState) -> GraphState:
        true_yields = state.global_attrs.get("current_true_expected_crop_yields")
        if true_yields is None:
            round_num = state.global_attrs.get("round_num", 0)
            # This error indicates an issue earlier in the pipeline (e.g., housekeeping)
            print(f"[ERROR] R{round_num} in create_stable_prediction_signal_transform: "
                  f"'current_true_expected_crop_yields' not found. Defaulting to empty array for signals.")
            true_yields = jnp.array([], dtype=jnp.float32)
            
        num_agents = state.num_nodes
        agent_specific_signals = {} # Dict[int, jnp.ndarray]

        # For a "stable" system with perfect information, all agents see the true yields.
        for i in range(num_agents):
            agent_specific_signals[i] = true_yields
        
        # If all agents have perfect signals, the market consensus is also the true yields.
        market_consensus = true_yields

        new_global_attrs = state.global_attrs.copy()
        new_global_attrs["agent_specific_prediction_signals"] = agent_specific_signals
        new_global_attrs["prediction_market_crop_signals"] = market_consensus # For compatibility
        return state.replace(global_attrs=new_global_attrs)
    return transform


# --- Helper for Performance History String ---
def _generate_performance_history_str(
    state: GraphState,
    agent_ids: List[int],
    max_history_items: int = 5
) -> str:
    """Generates a string summarizing performance history for given agent IDs."""
    history_lines = []
    cumulative_perf = state.node_attrs.get("cumulative_performance_score")
    num_decisions = state.node_attrs.get("num_decisions_made_history")

    if cumulative_perf is None or num_decisions is None:
        return "Performance history data is unavailable."

    for agent_id in agent_ids[:max_history_items]:
        avg_perf = (cumulative_perf[agent_id] / num_decisions[agent_id]) if num_decisions[agent_id] > 0 else 0.0
        history_lines.append(f"  - Agent {agent_id}: Avg Score {avg_perf:.2f} ({num_decisions[agent_id]} decisions)")
    
    if not history_lines:
        return "No performance history available for the listed agents."
    return "\n".join(history_lines)


# --- Agent Decision Transform (adapted for StablePortfolioDemocracyConfig) ---

def _get_llm_prompts_for_round(
    state: GraphState, 
    stable_sim_config: StablePortfolioDemocracyConfig,
    is_prd_election_voting_round: bool,
    portfolio_configs: List[PortfolioStrategyConfig],
    candidate_original_indices: jnp.ndarray,
    num_actual_candidates_for_election: int,
    pld_phase_specific_args: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], List[int]]:
    
    num_agents = state.num_nodes
    round_num = state.global_attrs.get("round_num", 0)
    agent_prompt_tasks: List[Dict[str, Any]] = []
    active_agents_for_llm: List[int] = []

    pld_phase_specific_args = pld_phase_specific_args or {}
    
    # === ROBUST PORTFOLIO OPTIONS STRING GENERATION ===
    portfolio_options_str = "No portfolios available."
    if portfolio_configs:
        try:
            agent_specific_signals = state.global_attrs.get("agent_specific_prediction_signals", {})
            # Use a representative agent's signals (agent 0) for portfolio option display
            representative_signals = agent_specific_signals.get(0, state.global_attrs.get("current_true_expected_crop_yields", jnp.array([])))
            
            if not isinstance(representative_signals, jnp.ndarray):
                representative_signals = jnp.array(representative_signals, dtype=jnp.float32)
            
            portfolio_expected_yields_str_list = []
            for p_idx, p_cfg in enumerate(portfolio_configs):
                p_weights = jnp.array(p_cfg.weights, dtype=jnp.float32)
                
                # Calculate expected yield
                if p_weights.size > 0 and representative_signals.size > 0 and p_weights.shape[0] == representative_signals.shape[0]:
                    expected_yield_val = float(jnp.sum(p_weights * representative_signals))
                    portfolio_expected_yields_str_list.append(
                        f"  {p_idx}: {p_cfg.name} (Expected Yield: {expected_yield_val:.3f}x)"
                    )
                else:
                    # Fallback for mismatched dimensions
                    portfolio_expected_yields_str_list.append(
                        f"  {p_idx}: {p_cfg.name} (Expected Yield: 1.000x)"
                    )
            
            if portfolio_expected_yields_str_list:
                portfolio_options_str = "\n".join(portfolio_expected_yields_str_list)
                if round_num < 3 or round_num % 20 == 0:
                    print(f"[DEBUG_PORTFOLIO_OPTIONS] R{round_num}: Generated {len(portfolio_configs)} portfolio options")
            else:
                print(f"[WARNING_PROMPT_GEN] R{round_num}: No portfolio options generated despite having {len(portfolio_configs)} configs")
                
        except Exception as e:
            print(f"[ERROR] R{round_num}: Failed to generate portfolio options: {e}")
            # Fallback generation
            portfolio_options_str = "\n".join([
                f"  {i}: Portfolio_{i} (Expected Yield: 1.000x)" 
                for i in range(len(portfolio_configs))
            ])

    # === FOR ANOMALY DETECTION LATER: Parse perceived EVs and store them ===
    agent_perceived_portfolio_evs_for_logging = {}
    if portfolio_configs:
        # Use representative_signals (agent 0 or true yields if stable) to parse "display" EVs
        agent_specific_signals = state.global_attrs.get("agent_specific_prediction_signals", {})
        representative_signals_for_ev_parsing = agent_specific_signals.get(0, state.global_attrs.get("current_true_expected_crop_yields", jnp.array([])))
        
        if not isinstance(representative_signals_for_ev_parsing, jnp.ndarray) or representative_signals_for_ev_parsing.size == 0:
            # Fallback if signals are not available for parsing display EVs
            if portfolio_configs and portfolio_configs[0].weights:
                num_crops_for_fallback = len(portfolio_configs[0].weights)
                representative_signals_for_ev_parsing = jnp.ones(num_crops_for_fallback)  # Assume neutral 1.0x
            else:
                representative_signals_for_ev_parsing = jnp.array([])

        if representative_signals_for_ev_parsing.size > 0:
            for p_idx_ev, p_cfg_ev in enumerate(portfolio_configs):
                p_weights_ev = jnp.array(p_cfg_ev.weights, dtype=jnp.float32)
                if p_weights_ev.shape[0] == representative_signals_for_ev_parsing.shape[0]:
                    expected_yield_val_ev = float(jnp.sum(p_weights_ev * representative_signals_for_ev_parsing))
                    agent_perceived_portfolio_evs_for_logging[p_idx_ev] = expected_yield_val_ev
                else:  # Fallback if shapes mismatch
                    agent_perceived_portfolio_evs_for_logging[p_idx_ev] = 1.0
    
    # === AGENT-SPECIFIC PROMPT GENERATION ===
    for i in range(num_agents):
        is_participating_this_round = bool(state.node_attrs["can_participate_this_round"][i])
        is_adversarial = bool(state.node_attrs["is_adversarial"][i])
        is_delegate_role_portfolio = bool(state.node_attrs["is_delegate"][i])

        if not is_participating_this_round:
            agent_prompt_tasks.append({
                "agent_id": i, "prompt": "NON_PARTICIPATING_NO_LLM_CALL", "max_tokens": 0, 
                "is_participating_this_round": False, "prompt_type": "none", 
                "mechanism": stable_sim_config.mechanism,
                "is_adversarial": is_adversarial, 
                "is_delegate_role_for_portfolio_vote": is_delegate_role_portfolio,
                # ADDED FOR ANOMALY LOGGING:
                "log_portfolio_options_str_seen": "NON_PARTICIPATING",
                "log_agent_perceived_portfolio_evs": {},
                "log_pld_delegate_performance_history_str": None,
            })
            continue
        
        active_agents_for_llm.append(i)
        
        # Base prompt arguments
        prompt_args: Dict[str, Any] = {
            "agent_id": i, "round_num": round_num,
            "is_delegate_role_for_portfolio_vote": is_delegate_role_portfolio,
            "is_adversarial": is_adversarial, 
            "mechanism": stable_sim_config.mechanism,
            "is_participating_this_round": True,
            "include_decision_framework": getattr(stable_sim_config, 'include_decision_framework_in_prompt', True),
            "portfolio_options_str": portfolio_options_str,  # Always provide portfolio options
        }

        # Determine prompt type and add specific arguments
        if is_prd_election_voting_round:
            current_agent_prompt_type = "election_vote"
            cand_opts_list = []
            if num_actual_candidates_for_election > 0:
                for cand_list_idx, original_agent_id_val in enumerate(candidate_original_indices.tolist()):
                    cand_opts_list.append(f"  {cand_list_idx}: Agent {original_agent_id_val}")
            prompt_args.update({
                "prd_candidate_options_str": "\n".join(cand_opts_list) if cand_opts_list else "No candidates running.",
                "prd_num_candidates": num_actual_candidates_for_election,
                "prd_candidate_performance_history_str": _generate_performance_history_str(state, candidate_original_indices.tolist())
            })
        elif stable_sim_config.mechanism == "PLD":
            current_agent_prompt_type = pld_phase_specific_args.get("pld_current_phase_prompt_type", "error_pld_phase_not_set")
            
            if current_agent_prompt_type == "pld_voter_delegation_choice":
                prompt_args.update({
                    "pld_declared_delegate_choices_str": pld_phase_specific_args.get("pld_declared_delegate_choices_str", "N/A"),
                    "pld_performance_history_str": pld_phase_specific_args.get("pld_delegate_performance_history_str", "N/A")
                })
            elif current_agent_prompt_type == "pld_final_power_holder_vote":
                prompt_args.update({
                    "pld_agent_voting_power_str": f"{state.node_attrs['voting_power'][i]:.2f}" if "voting_power" in state.node_attrs else "N/A",
                    "pld_prior_declaration_reminder_str": pld_phase_specific_args.get(f"pld_prior_declaration_agent_{i}", "")
                })
        else:
            # PDD/PRD portfolio vote
            current_agent_prompt_type = "portfolio_vote"

        prompt_args["prompt_type"] = current_agent_prompt_type
        
        # Generate the prompt
        try:
            prompt_result = stable_sim_config.prompt_settings.generate_prompt(**prompt_args)
            agent_prompt_tasks.append({
                "agent_id": i, 
                "prompt": prompt_result["prompt"], 
                "max_tokens": prompt_result["max_tokens"],
                "is_participating_this_round": True, 
                "prompt_type": current_agent_prompt_type,
                "is_adversarial": is_adversarial, 
                "is_delegate_role_for_portfolio_vote": is_delegate_role_portfolio,
                "mechanism": stable_sim_config.mechanism,
                # ADDED FOR ANOMALY LOGGING:
                "log_portfolio_options_str_seen": portfolio_options_str,
                "log_agent_perceived_portfolio_evs": agent_perceived_portfolio_evs_for_logging,
                "log_pld_delegate_performance_history_str": pld_phase_specific_args.get("pld_delegate_performance_history_str") if stable_sim_config.mechanism == "PLD" and current_agent_prompt_type == "pld_voter_delegation_choice" else None,
            })
        except Exception as e:
            print(f"[ERROR] R{round_num} A{i}: Failed to generate prompt: {e}")
            # Fallback task
            agent_prompt_tasks.append({
                "agent_id": i, "prompt": "ERROR_IN_PROMPT_GENERATION", "max_tokens": 50,
                "is_participating_this_round": False, "prompt_type": "error",
                "is_adversarial": is_adversarial, "is_delegate_role_for_portfolio_vote": is_delegate_role_portfolio,
                "mechanism": stable_sim_config.mechanism,
                # ADDED FOR ANOMALY LOGGING:
                "log_portfolio_options_str_seen": "ERROR_IN_PROMPT_GENERATION",
                "log_agent_perceived_portfolio_evs": {},
                "log_pld_delegate_performance_history_str": None,
            })
        
    return agent_prompt_tasks, active_agents_for_llm

# --- Helper Function 2: Execute LLM Calls ---
def _execute_llm_calls_parallel(
    llm_service: Optional[LLMService],
    agent_prompt_tasks: List[Dict[str, Any]], # Full list from _get_llm_prompts_for_round
    active_agents_for_llm: List[int], # List of agent_ids that need LLM calls
    round_num: int, # For logging
    max_workers_config: int
) -> Dict[int, str]:
    llm_responses: Dict[int, str] = {}
    if not llm_service or not active_agents_for_llm:
        if active_agents_for_llm: # Active agents but no LLM service
            print(f"[LLM_CALLS] R{round_num}: No LLM service. Active agents ({len(active_agents_for_llm)}) will use fallback.")
        for agent_id in active_agents_for_llm: # Ensure entries for fallback
            llm_responses[agent_id] = ""
        return llm_responses

    actual_max_workers = min(len(active_agents_for_llm), max_workers_config)
    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        future_to_agent_id_map = {}
        for task in agent_prompt_tasks: # Iterate all tasks
            if task["agent_id"] in active_agents_for_llm: # Only submit if agent is active
                 future_to_agent_id_map[executor.submit(llm_service.generate, task["prompt"], task["max_tokens"])] = task["agent_id"]
        
        for future in as_completed(future_to_agent_id_map):
            agent_id_completed = future_to_agent_id_map[future]
            try:
                llm_responses[agent_id_completed] = future.result()
            except Exception as e_llm:
                print(f"[LLM_CALL_ERROR] R{round_num} A{agent_id_completed}: {e_llm}")
                llm_responses[agent_id_completed] = ""
    return llm_responses


# --- Helper Function 3: Parse LLM Responses and Prepare Node Attribute Updates ---
def _parse_llm_responses_and_prepare_updates(
    state: GraphState,
    stable_sim_config: StablePortfolioDemocracyConfig,
    llm_responses: Dict[int, str],
    agent_prompt_tasks: List[Dict[str, Any]],
    is_prd_election_voting_round: bool,
    portfolio_configs: List[PortfolioStrategyConfig],
    num_actual_candidates_for_election: int,
    new_agent_portfolio_votes: jnp.ndarray,
    new_delegation_target: jnp.ndarray,
    new_agent_election_candidate_votes: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[str], List[Dict[str, Any]]]:  # Added List for anomalies
    
    num_agents = state.num_nodes
    round_num = state.global_attrs.get("round_num", 0)
    num_portfolios = len(portfolio_configs)
    decision_summary_log: List[str] = []
    current_round_anomalies: List[Dict[str, Any]] = []  # NEW: Store anomalies

    if round_num < 3 or round_num % 20 == 0:
        print(f"[DEBUG_PARSE_START] R{round_num}: Parsing {len(agent_prompt_tasks)} agent responses for {num_portfolios} portfolios")

    for task_data in agent_prompt_tasks:
        agent_idx = task_data["agent_id"]
        log_prefix = f"A{agent_idx}({'Adv' if task_data['is_adversarial'] else 'Aln'}{'Del' if task_data['is_delegate_role_for_portfolio_vote'] else 'Vot'})"

        if not task_data["is_participating_this_round"]:
            # Handle non-participation
            decision_summary_log.append(f"{log_prefix}→NonPart")
            continue

        llm_response = llm_responses.get(agent_idx, "").strip()
        action_successfully_parsed = False
        
        if round_num < 3 or round_num % 20 == 0:
            print(f"[DEBUG_PARSE_AGENT] R{round_num} {log_prefix} ({task_data['prompt_type']}): Response length {len(llm_response)}, first 100 chars: '{llm_response[:100]}'"
                  + ("..." if len(llm_response) > 100 else ""))

        # Initialize decision variables
        current_agent_delegation_choice = -1
        current_agent_portfolio_approvals: List[int] = []
        pld_voter_action_choice = None

        # === PARSING LOGIC BY PROMPT TYPE ===
        try:
            if task_data["prompt_type"] == "election_vote":
                if num_actual_candidates_for_election == 0:
                    action_successfully_parsed = True
                    decision_summary_log.append(f"{log_prefix}→NoCandToVote")
                else:
                    cand_votes_match = re.search(r"CandidateVotes\s*:\s*\[([^\]]*)\]", llm_response, re.IGNORECASE | re.MULTILINE)
                    if cand_votes_match:
                        cand_vote_content = cand_votes_match.group(1).strip()
                        if not cand_vote_content:  # Empty brackets
                            parsed_cand_approval_list = [0] * num_actual_candidates_for_election
                            if round_num < 3 or round_num % 20 == 0:
                                print(f"[DEBUG_PARSE_ELECTION] R{round_num} {log_prefix}: Empty brackets, defaulting to all zeros")
                        else:
                            cand_vote_list_str = [v.strip() for v in cand_vote_content.split(',')]
                            parsed_cand_approval_list = []
                            for v_s in cand_vote_list_str:
                                if v_s in ['1', '1.0']: 
                                    parsed_cand_approval_list.append(1)
                                elif v_s in ['0', '0.0']: 
                                    parsed_cand_approval_list.append(0)
                                else:
                                    if round_num < 3 or round_num % 20 == 0:
                                        print(f"[DEBUG_PARSE_ELECTION_INVALID] R{round_num} {log_prefix}: Invalid vote value '{v_s}'")
                                    parsed_cand_approval_list.append(0)  # Default to 0
                        
                        # Ensure correct length
                        if len(parsed_cand_approval_list) != num_actual_candidates_for_election:
                            if round_num < 3 or round_num % 20 == 0:
                                print(f"[DEBUG_PARSE_ELECTION_LENGTH] R{round_num} {log_prefix}: Got {len(parsed_cand_approval_list)} votes, expected {num_actual_candidates_for_election}. Adjusting.")
                            # Pad or truncate to correct length
                            while len(parsed_cand_approval_list) < num_actual_candidates_for_election:
                                parsed_cand_approval_list.append(0)
                            parsed_cand_approval_list = parsed_cand_approval_list[:num_actual_candidates_for_election]
                        
                        new_agent_election_candidate_votes = new_agent_election_candidate_votes.at[agent_idx, :].set(
                            jnp.array(parsed_cand_approval_list, dtype=jnp.int32)
                        )
                        action_successfully_parsed = True
                        decision_summary_log.append(f"{log_prefix}→CandVotes:{sum(parsed_cand_approval_list)}aps")
                    else:
                        if round_num < 3 or round_num % 20 == 0:
                            print(f"[DEBUG_PARSE_ELECTION_NO_MATCH] R{round_num} {log_prefix}: No CandidateVotes pattern found")

            elif task_data["prompt_type"] == "pld_voter_delegation_choice":
                pld_action_match = re.search(r"Action:\s*(\w+)", llm_response, re.IGNORECASE | re.MULTILINE)
                if pld_action_match:
                    action_text = pld_action_match.group(1).upper().replace("_", "")  # Handle VOTE_DIRECTLY
                    if "DELEGATE" in action_text:
                        pld_target_match = re.search(r"AgentID:\s*(\d+)", llm_response, re.IGNORECASE | re.MULTILINE)
                        if pld_target_match:
                            raw_parsed_target_id = int(pld_target_match.group(1))
                            if (0 <= raw_parsed_target_id < num_agents and 
                                raw_parsed_target_id != agent_idx and 
                                state.node_attrs["is_delegate"][raw_parsed_target_id]):
                                current_agent_delegation_choice = raw_parsed_target_id
                                pld_voter_action_choice = "DELEGATE"
                                action_successfully_parsed = True
                                decision_summary_log.append(f"{log_prefix}→DelegateTo:{current_agent_delegation_choice}")
                            else: 
                                if round_num < 3 or round_num % 20 == 0:
                                    print(f"[DEBUG_PARSE_PLD_INVALID_TARGET] R{round_num} {log_prefix}: Invalid target {raw_parsed_target_id}")
                        else:
                            if round_num < 3 or round_num % 20 == 0:
                                print(f"[DEBUG_PARSE_PLD_NO_AGENT_ID] R{round_num} {log_prefix}: DELEGATE action but no AgentID found")
                    elif "VOTE" in action_text or "DIRECT" in action_text:
                        pld_voter_action_choice = "VOTE_DIRECTLY"
                        action_successfully_parsed = True
                        decision_summary_log.append(f"{log_prefix}→VoteDirectIntent")
                    else:
                        if round_num < 3 or round_num % 20 == 0:
                            print(f"[DEBUG_PARSE_PLD_UNKNOWN_ACTION] R{round_num} {log_prefix}: Unknown action '{action_text}'")
                else:
                    if round_num < 3 or round_num % 20 == 0:
                        print(f"[DEBUG_PARSE_PLD_NO_ACTION] R{round_num} {log_prefix}: No Action pattern found")

            elif task_data["prompt_type"] in ["portfolio_vote", "pld_delegate_declaration", "pld_final_power_holder_vote"]:
                if num_portfolios > 0:
                    vote_keyword = "DeclaredVotes" if task_data["prompt_type"] == "pld_delegate_declaration" else "Votes"
                    portfolio_votes_match = re.search(rf"{vote_keyword}\s*:\s*\[([^\]]*)\]", llm_response, re.IGNORECASE | re.MULTILINE)
                    
                    if portfolio_votes_match:
                        vote_content_str = portfolio_votes_match.group(1).strip()
                        if round_num < 3 or round_num % 20 == 0:
                            print(f"[DEBUG_PARSE_PORTFOLIO] R{round_num} {log_prefix}: Found {vote_keyword} content: '{vote_content_str}'")
                        
                        if not vote_content_str:  # Empty brackets
                            if round_num < 3 or round_num % 20 == 0:
                                print(f"[DEBUG_PARSE_PORTFOLIO_EMPTY] R{round_num} {log_prefix}: Empty brackets - treating as all zeros")
                            current_agent_portfolio_approvals = []  # Will result in all zeros
                            action_successfully_parsed = True
                            decision_summary_log.append(f"{log_prefix}→PVotes:0aps (empty brackets)")
                        else:
                            vote_list_str_raw = [v.strip() for v in vote_content_str.split(',')]
                            parsed_portfolio_approvals = []

                            for idx_loop, vote_char in enumerate(vote_list_str_raw):
                                if idx_loop >= num_portfolios:
                                    break  # Don't process more votes than portfolios
                                    
                                if vote_char in ['1', '1.0']:
                                    parsed_portfolio_approvals.append(idx_loop)
                                elif vote_char in ['0', '0.0']:
                                    pass  # Not approved
                                else:
                                    if round_num < 3 or round_num % 20 == 0:
                                        print(f"[DEBUG_PARSE_PORTFOLIO_INVALID] R{round_num} {log_prefix}: Invalid vote '{vote_char}' at index {idx_loop}, treating as 0")
                            
                            current_agent_portfolio_approvals = parsed_portfolio_approvals
                            action_successfully_parsed = True
                            decision_summary_log.append(f"{log_prefix}→PVotes:{len(current_agent_portfolio_approvals)}aps")
                            
                            if len(vote_list_str_raw) != num_portfolios:
                                if round_num < 3 or round_num % 20 == 0:
                                    print(f"[DEBUG_PARSE_PORTFOLIO_LENGTH] R{round_num} {log_prefix}: Got {len(vote_list_str_raw)} votes, expected {num_portfolios}")
                    else:
                        if round_num < 3 or round_num % 20 == 0:
                            print(f"[DEBUG_PARSE_PORTFOLIO_NO_MATCH] R{round_num} {log_prefix}: No {vote_keyword} pattern found")
                else:
                    if round_num < 3 or round_num % 20 == 0:
                        print(f"[DEBUG_PARSE_PORTFOLIO_NO_PORTFOLIOS] R{round_num} {log_prefix}: No portfolios to vote on")
                    action_successfully_parsed = True
                    decision_summary_log.append(f"{log_prefix}→NoPortfolios")
        
        except Exception as e:
            print(f"[DEBUG_PARSE_ERROR] R{round_num} {log_prefix}: Parsing exception: {e}")
            action_successfully_parsed = False

        # === ANOMALY DETECTION LOGIC ===
        if task_data["is_adversarial"]:  # Only check anomalies for adversarial agents
            
            # Anomaly 1: Adversarial PLD agent delegates to an ALIGNED agent
            if task_data["mechanism"] == "PLD" and \
               task_data["prompt_type"] == "pld_voter_delegation_choice" and \
               pld_voter_action_choice == "DELEGATE" and current_agent_delegation_choice != -1:
                
                chosen_delegate_id = current_agent_delegation_choice
                if 0 <= chosen_delegate_id < num_agents:
                    chosen_delegate_is_adversarial = bool(state.node_attrs["is_adversarial"][chosen_delegate_id])
                    if not chosen_delegate_is_adversarial:  # Delegated to ALIGNED!
                        current_round_anomalies.append({
                            "anomaly_type": "ADV_DELEGATES_TO_ALIGNED",
                            "round_num": round_num,
                            "agent_id": agent_idx,
                            "mechanism": task_data["mechanism"],
                            "chosen_delegate_id": chosen_delegate_id,
                            "raw_llm_prompt": task_data["prompt"],
                            "raw_llm_response": llm_response,
                            "log_pld_delegate_performance_history_str": task_data.get("log_pld_delegate_performance_history_str")
                        })

            # Anomaly 2: Adversarial agent votes for a "good" portfolio
            if task_data["prompt_type"] in ["portfolio_vote", "pld_delegate_declaration", "pld_final_power_holder_vote"]:
                agent_perceived_evs = task_data.get("log_agent_perceived_portfolio_evs", {})
                if agent_perceived_evs and current_agent_portfolio_approvals is not None:
                    
                    if not agent_perceived_evs:
                        perceived_worst_ev = 0.0
                        perceived_best_ev = 1.0
                    else:
                        perceived_worst_ev = min(agent_perceived_evs.values())
                        perceived_best_ev = max(agent_perceived_evs.values())

                    for approved_idx in current_agent_portfolio_approvals:
                        if approved_idx in agent_perceived_evs:
                            chosen_portfolio_ev = agent_perceived_evs[approved_idx]
                            
                            # Define "anomalous good choice" for an adversarial agent
                            is_anomalous_choice = False
                            anomaly_threshold = 1.01  # If they perceive it as slightly profitable
                            if chosen_portfolio_ev > anomaly_threshold:
                                is_anomalous_choice = True
                            elif perceived_best_ev > perceived_worst_ev:
                                if chosen_portfolio_ev > (perceived_worst_ev + (perceived_best_ev - perceived_worst_ev) * 0.25):
                                    is_anomalous_choice = True
                            
                            if is_anomalous_choice:
                                current_round_anomalies.append({
                                    "anomaly_type": "ADV_VOTES_FOR_NON_WORST_EV",
                                    "round_num": round_num,
                                    "agent_id": agent_idx,
                                    "mechanism": task_data["mechanism"],
                                    "prompt_type": task_data["prompt_type"],
                                    "approved_portfolio_idx": approved_idx,
                                    "chosen_portfolio_perceived_ev": chosen_portfolio_ev,
                                    "agent_perceived_evs_str": str(agent_perceived_evs),
                                    "raw_llm_prompt": task_data["prompt"],
                                    "raw_llm_response": llm_response,
                                })

        # === APPLY PARSED RESULTS TO ARRAYS ===
        try:
            if task_data["mechanism"] == "PLD" and task_data["prompt_type"] == "pld_voter_delegation_choice":
                if pld_voter_action_choice == "DELEGATE" and current_agent_delegation_choice != -1:
                    new_delegation_target = new_delegation_target.at[agent_idx].set(current_agent_delegation_choice)
                    # Clear portfolio votes for delegating voters
                    if num_portfolios > 0 and new_agent_portfolio_votes.shape[1] > 0: 
                        new_agent_portfolio_votes = new_agent_portfolio_votes.at[agent_idx, :].set(0)
                elif pld_voter_action_choice == "VOTE_DIRECTLY":
                    new_delegation_target = new_delegation_target.at[agent_idx].set(-1)
            
            elif task_data["prompt_type"] in ["portfolio_vote", "pld_delegate_declaration", "pld_final_power_holder_vote"]:
                if num_portfolios > 0 and new_agent_portfolio_votes.shape[1] > 0:
                    # Create vote array from parsed approvals
                    agent_pf_vote_arr = jnp.zeros(num_portfolios, dtype=jnp.int32)
                    if current_agent_portfolio_approvals:  # Non-empty list of approved indices
                        valid_indices = [idx for idx in current_agent_portfolio_approvals if 0 <= idx < num_portfolios]
                        if valid_indices:
                            agent_pf_vote_arr = agent_pf_vote_arr.at[jnp.array(valid_indices, dtype=jnp.int32)].set(1)
                    
                    new_agent_portfolio_votes = new_agent_portfolio_votes.at[agent_idx, :].set(agent_pf_vote_arr)
                    if round_num < 3 or round_num % 20 == 0:
                        print(f"[DEBUG_APPLY_PORTFOLIO] R{round_num} {log_prefix}: Applied approvals {current_agent_portfolio_approvals} -> array {agent_pf_vote_arr}")
                    
                    # For PLD delegate declaration, ensure they are not marked as delegating
                    if task_data["mechanism"] == "PLD" and task_data["prompt_type"] == "pld_delegate_declaration":
                        new_delegation_target = new_delegation_target.at[agent_idx].set(-1)
        
        except Exception as e:
            print(f"[DEBUG_APPLY_ERROR] R{round_num} {log_prefix}: Array application exception: {e}")
            action_successfully_parsed = False

        # === HANDLE PARSING FAILURES ===
        if not action_successfully_parsed:
            print(f"[DEBUG_FALLBACK] R{round_num} {log_prefix} ({task_data['mechanism']}, {task_data['prompt_type']}): Parse failed. Full response: '{llm_response}'")
            decision_summary_log.append(f"{log_prefix}→ParseFAIL")
            
            # Apply appropriate fallbacks
            if task_data["prompt_type"] in ["portfolio_vote", "pld_delegate_declaration", "pld_final_power_holder_vote"] and num_portfolios > 0:
                fallback_zeros = jnp.zeros(num_portfolios, dtype=jnp.int32)
                new_agent_portfolio_votes = new_agent_portfolio_votes.at[agent_idx, :].set(fallback_zeros)
                if round_num < 3 or round_num % 20 == 0:
                    print(f"[DEBUG_FALLBACK_APPLIED] R{round_num} {log_prefix}: Applied portfolio fallback zeros")

    if round_num < 3 or round_num % 20 == 0:
        print(f"[DEBUG_PARSE_END] R{round_num}: Completed parsing. Summary: {len([log for log in decision_summary_log if 'ParseFAIL' not in log])}/{len(decision_summary_log)} successful")
    
    return new_agent_portfolio_votes, new_delegation_target, new_agent_election_candidate_votes, decision_summary_log, current_round_anomalies


# --- Main Creator Function ---
def create_llm_agent_decision_transform_stable(
    llm_service: Optional[LLMService],
    stable_sim_config: StablePortfolioDemocracyConfig
) -> Transform:
    def transform(state: GraphState) -> GraphState:
        round_num = state.global_attrs.get("round_num", 0)
        num_agents = state.num_nodes

        new_global_attrs = state.global_attrs.copy() # Start with a copy for global updates
        final_node_attrs = state.node_attrs.copy()   # Start with a copy for node updates

        # Initialize/ensure PLD specific node attributes if mechanism is PLD
        if stable_sim_config.mechanism == "PLD" and "pld_intends_to_vote_directly" not in final_node_attrs:
            final_node_attrs["pld_intends_to_vote_directly"] = jnp.zeros(num_agents, dtype=jnp.bool_)

        # --- Common Preparations ---

        is_prd_election_voting_round = False
        if stable_sim_config.mechanism == "PRD" and state.global_attrs.get("rounds_until_next_election_prd", 0) == 0:
            is_prd_election_voting_round = True
            print(f"[LLM_DECISION_STABLE] R{round_num}: PRD Election Voting Round. Prompting for candidate votes.")
        
        portfolio_configs_generic = state.global_attrs.get("portfolio_configs", [])
        portfolio_configs_typed: List[PortfolioStrategyConfig] = [] # Ensure it's typed
        if portfolio_configs_generic:
            first_p_item = portfolio_configs_generic[0]
            if isinstance(first_p_item, PortfolioStrategyConfig): 
                portfolio_configs_typed = portfolio_configs_generic
            elif hasattr(first_p_item, '__dict__'): 
                portfolio_configs_typed = [PortfolioStrategyConfig(**pcfg.__dict__) for pcfg in portfolio_configs_generic]
            else: 
                portfolio_configs_typed = [PortfolioStrategyConfig(**pcfg) for pcfg in portfolio_configs_generic]

        candidate_flag_attr = getattr(stable_sim_config, 'prd_candidate_flag_attr', "is_delegate")
        candidate_mask = state.node_attrs.get(candidate_flag_attr, jnp.zeros(num_agents, dtype=jnp.bool_))
        candidate_original_indices = jnp.where(candidate_mask)[0]
        num_actual_candidates = candidate_original_indices.shape[0]

        # Prepare initial arrays for updates
        num_portfolios_init = len(portfolio_configs_typed)
        current_agent_portfolio_votes = final_node_attrs.get("agent_portfolio_votes", jnp.zeros((num_agents, max(0,num_portfolios_init) ), dtype=jnp.int32)).copy()
        current_delegation_target = final_node_attrs.get("delegation_target", -jnp.ones(num_agents, dtype=jnp.int32)) # No .copy() needed if directly assigned
        
        initial_election_votes_shape_cols = num_actual_candidates if num_actual_candidates > 0 else 0
        current_election_votes = final_node_attrs.get("agent_votes_for_election_candidates", jnp.zeros((num_agents, max(0,initial_election_votes_shape_cols)), dtype=jnp.int32)).copy()

        if current_election_votes.shape[1] != initial_election_votes_shape_cols:
            current_election_votes = jnp.zeros((num_agents, initial_election_votes_shape_cols), dtype=jnp.int32)

        decision_summary_phase1: List[str] = []
        all_round_anomalies: List[Dict[str, Any]] = []  # Initialize anomalies list
        
        # --- Multi-Phase Logic: PLD has 3 phases, others have 1 phase ---
        if stable_sim_config.mechanism == "PLD" and not is_prd_election_voting_round:
            # --- PLD Phase 1: Delegate Declaration ---
            if round_num < 3 or round_num % 20 == 0:
                print(f"[LLM_DECISION_STABLE] R{round_num}: PLD Phase 1 - Delegate Portfolio Declaration.")
            delegate_mask_pld_phase1 = final_node_attrs["is_delegate"] & final_node_attrs["can_participate_this_round"]
            # Create a temporary state view for this phase's prompting
            temp_node_attrs_pld1 = final_node_attrs.copy()
            temp_node_attrs_pld1["can_participate_this_round"] = delegate_mask_pld_phase1
            temp_state_pld_phase1 = state.replace(node_attrs=temp_node_attrs_pld1)
            
            agent_prompt_tasks_pld1, active_delegates_pld1 = _get_llm_prompts_for_round(
                temp_state_pld_phase1, stable_sim_config, 
                is_prd_election_voting_round=False, 
                portfolio_configs=portfolio_configs_typed, 
                candidate_original_indices=jnp.array([]), num_actual_candidates_for_election=0,
                pld_phase_specific_args={"pld_current_phase_prompt_type": "pld_delegate_declaration"}
            )
            llm_responses_pld1 = _execute_llm_calls_parallel(llm_service, agent_prompt_tasks_pld1, active_delegates_pld1, round_num, getattr(stable_sim_config, 'max_llm_workers', 8))
            
            current_agent_portfolio_votes, current_delegation_target, _, decision_summary_phase1, round_anomalies_phase1 = \
                _parse_llm_responses_and_prepare_updates(
                    temp_state_pld_phase1, stable_sim_config, llm_responses_pld1, agent_prompt_tasks_pld1,
                    is_prd_election_voting_round=False, portfolio_configs=portfolio_configs_typed, num_actual_candidates_for_election=0,
                    new_agent_portfolio_votes=current_agent_portfolio_votes, new_delegation_target=current_delegation_target,
                    new_agent_election_candidate_votes=current_election_votes
                )
            
            if decision_summary_phase1 and (round_num < 3 or round_num % 20 == 0):
                print(f"[R{round_num:02d}] PLD Phase 1 Decisions: {' | '.join(decision_summary_phase1)}")
            final_node_attrs["agent_portfolio_votes"] = current_agent_portfolio_votes # Store declarations
            final_node_attrs["delegation_target"] = current_delegation_target

            # Prepare delegate declarations string for Phase 2 prompts
            pld_delegate_declarations_for_phase2_prompt = {}
            for del_idx in range(num_agents):
                if delegate_mask_pld_phase1[del_idx]:
                    del_votes_declared = current_agent_portfolio_votes[del_idx, :]
                    approved_indices = jnp.where(del_votes_declared == 1)[0].tolist()
                    pld_delegate_declarations_for_phase2_prompt[del_idx] = approved_indices
            new_global_attrs["pld_delegate_declarations_cache_REF"] = pld_delegate_declarations_for_phase2_prompt # Store for Phase 2 & 3

            # --- PLD Phase 2: Voter Delegation/Intent to Vote ---
            if round_num < 3 or round_num % 20 == 0:
                print(f"[LLM_DECISION_STABLE] R{round_num}: PLD Phase 2 - Voter Delegation/Intent to Vote.")
            voter_mask_pld_phase2 = (~final_node_attrs["is_delegate"]) & final_node_attrs["can_participate_this_round"]
            temp_node_attrs_pld2 = final_node_attrs.copy()
            temp_node_attrs_pld2["can_participate_this_round"] = voter_mask_pld_phase2
            temp_state_pld_phase2 = state.replace(node_attrs=temp_node_attrs_pld2, global_attrs=new_global_attrs)
            
            declared_choices_str_list = []
            for del_id, choices in pld_delegate_declarations_for_phase2_prompt.items():
                declared_choices_str_list.append(f"  - Delegate {del_id} declared approvals for portfolios: {choices if choices else 'None'}")
            pld_declared_delegate_choices_str_for_voters_prompt = "\n".join(declared_choices_str_list) if declared_choices_str_list else "No delegates made declarations."

            # Get performance history for original delegates
            original_delegate_indices = jnp.where(final_node_attrs["is_delegate"])[0].tolist()
            pld_delegate_performance_history_str = _generate_performance_history_str(state, original_delegate_indices)

            agent_prompt_tasks_pld2, active_voters_pld2 = _get_llm_prompts_for_round(
                temp_state_pld_phase2, stable_sim_config,
                is_prd_election_voting_round=False, 
                portfolio_configs=portfolio_configs_typed,
                candidate_original_indices=jnp.array([]), num_actual_candidates_for_election=0,
                pld_phase_specific_args={
                    "pld_current_phase_prompt_type": "pld_voter_delegation_choice",
                    "pld_declared_delegate_choices_str": pld_declared_delegate_choices_str_for_voters_prompt,
                    "pld_delegate_performance_history_str": pld_delegate_performance_history_str
                }
            )
            llm_responses_pld2 = _execute_llm_calls_parallel(llm_service, agent_prompt_tasks_pld2, active_voters_pld2, round_num, getattr(stable_sim_config, 'max_llm_workers', 8))
            
            _, current_delegation_target, _, decision_summary_phase1_p2, round_anomalies_phase2 = \
                _parse_llm_responses_and_prepare_updates( # Reusing decision_summary_phase1 for combined log
                    temp_state_pld_phase2, stable_sim_config, llm_responses_pld2, agent_prompt_tasks_pld2,
                    is_prd_election_voting_round=False, portfolio_configs=portfolio_configs_typed, num_actual_candidates_for_election=0,
                    new_agent_portfolio_votes=current_agent_portfolio_votes, # Not updated by voters in Phase 2
                    new_delegation_target=current_delegation_target,
                    new_agent_election_candidate_votes=current_election_votes
                )
            
            # Combine anomalies from both phases
            all_round_anomalies = round_anomalies_phase1 + round_anomalies_phase2
            decision_summary_phase1.extend(decision_summary_phase1_p2) # Combine logs
            if decision_summary_phase1_p2 and (round_num < 3 or round_num % 20 == 0):
                 print(f"[R{round_num:02d}] PLD Phase 2 Decisions: {' | '.join(decision_summary_phase1_p2)}")
            final_node_attrs["delegation_target"] = current_delegation_target

            # Update pld_intends_to_vote_directly based on delegation target from Phase 2 parsing
            temp_pld_intends_vote = final_node_attrs["pld_intends_to_vote_directly"]
            for task_data in agent_prompt_tasks_pld2:
                agent_id_p2 = task_data["agent_id"]
                if voter_mask_pld_phase2[agent_id_p2]: # If this voter was active in phase 2
                    # If they didn't delegate (target is -1), they intend to vote directly.
                    temp_pld_intends_vote = temp_pld_intends_vote.at[agent_id_p2].set(current_delegation_target[agent_id_p2] == -1)
            final_node_attrs["pld_intends_to_vote_directly"] = temp_pld_intends_vote

        else: # PDD, PRD, or PRD Election Round (Standard single-phase decision making)
            agent_prompt_tasks, active_agents_for_llm = _get_llm_prompts_for_round(
                state, stable_sim_config, is_prd_election_voting_round,
                portfolio_configs_typed, candidate_original_indices, num_actual_candidates
            )
            llm_responses = _execute_llm_calls_parallel(
                llm_service, agent_prompt_tasks, active_agents_for_llm, round_num,
                getattr(stable_sim_config, 'max_llm_workers', 8)
            )
            current_agent_portfolio_votes, current_delegation_target, current_election_votes, decision_summary_phase1, all_round_anomalies = \
                _parse_llm_responses_and_prepare_updates(
                    state, stable_sim_config, llm_responses, agent_prompt_tasks,
                    is_prd_election_voting_round, portfolio_configs_typed, num_actual_candidates,
                    current_agent_portfolio_votes, current_delegation_target, current_election_votes
                )

        # Logging for main decision phase (PLD Phase 1&2 combined, or PDD/PRD)
        if decision_summary_phase1:
            if round_num < 3 or round_num % 20 == 0: # Detailed summary less often
                 print(f"[R{round_num:02d}] Main Decision Phase Summary: {' | '.join(decision_summary_phase1)}")
            elif round_num % 5 == 0 : # Concise summary more often (but still less than every round)
                 print(f"[R{round_num:02d}] Main Decisions processed for {len(decision_summary_phase1)} agents.")
        
        # Prepare final node attributes for state update
        # For PLD, agent_portfolio_votes from Phase 1 (declarations) are stored.
        # For PDD/PRD, actual portfolio votes are stored.
        # This will be used by PLD Phase 3 or by the main portfolio_voting_transform.
        if not (stable_sim_config.mechanism == "PLD" and not is_prd_election_voting_round): # Only update if not PLD main phase
            final_node_attrs["agent_portfolio_votes"] = current_agent_portfolio_votes
        if stable_sim_config.mechanism == "PLD":
            pass # delegation_target and pld_intends_to_vote_directly already updated in final_node_attrs
        if is_prd_election_voting_round: # Only update if it was an election round
            final_node_attrs["agent_votes_for_election_candidates"] = current_election_votes
        
        # Set flag for portfolio resolution gating
        portfolio_decision_active = True
        if (is_prd_election_voting_round and getattr(stable_sim_config, "prd_election_round_is_exclusive", True)) or \
           (stable_sim_config.mechanism == "PLD" and not is_prd_election_voting_round) : # Defer portfolio decision for PLD until Phase 3
            portfolio_decision_active = False
        
        new_global_attrs["portfolio_decision_active_this_round"] = portfolio_decision_active
        
        # Store anomalies in global_attrs
        new_global_attrs["round_anomaly_log"] = all_round_anomalies

        return state.replace(node_attrs=final_node_attrs, global_attrs=new_global_attrs)
        
    return transform


# --- PLD Phase 3: Final Power Holder Vote Transform ---
def create_pld_final_power_holder_vote_transform(
    llm_service: Optional[LLMService],
    stable_sim_config: StablePortfolioDemocracyConfig
) -> Transform:
    """
    PLD Phase 3: Final portfolio voting by agents who hold voting power after delegation.
    FIXED: Only enforce delegate consistency if their Phase 3 response is invalid.
    """
    def transform(state: GraphState) -> GraphState:
        round_num = state.global_attrs.get("round_num", 0)
        num_agents = state.num_nodes

        if stable_sim_config.mechanism != "PLD":
            return state

        if round_num < 3 or round_num % 20 == 0:
            print(f"[PLD_PHASE_3_VOTE] R{round_num}: Final Power Holder Vote.")
        # Determine eligibility
        voting_power = state.node_attrs.get("voting_power", jnp.zeros(num_agents))
        pld_intends_to_vote_directly = state.node_attrs.get("pld_intends_to_vote_directly", jnp.zeros(num_agents, dtype=jnp.bool_))
        is_delegate_role = state.node_attrs.get("is_delegate", jnp.zeros(num_agents, dtype=jnp.bool_))
        
        eligible_final_voters_mask = ((is_delegate_role & (voting_power > 0)) | 
                                      (pld_intends_to_vote_directly & (voting_power > 0)))

        eligible_voters_count = jnp.sum(eligible_final_voters_mask)
        if eligible_voters_count == 0:
            if round_num < 3 or round_num % 20 == 0:
                print(f"[PLD_PHASE_3_VOTE] R{round_num}: No eligible power holders for final vote.")
            new_global_attrs = state.global_attrs.copy()
            new_global_attrs["portfolio_decision_active_this_round"] = True
            new_global_attrs["round_anomaly_log"] = []  # No anomalies if no voters
            return state.replace(global_attrs=new_global_attrs)

        if round_num < 3 or round_num % 20 == 0:
            print(f"[PLD_PHASE_3_VOTE] R{round_num}: {eligible_voters_count} agents eligible for final power holder vote.")

        # Get portfolio configs
        portfolio_configs_generic = state.global_attrs.get("portfolio_configs", [])
        portfolio_configs_typed: List[PortfolioStrategyConfig] = []
        if portfolio_configs_generic:
            first_p_item = portfolio_configs_generic[0]
            if isinstance(first_p_item, PortfolioStrategyConfig): 
                portfolio_configs_typed = portfolio_configs_generic
            elif hasattr(first_p_item, '__dict__'): 
                portfolio_configs_typed = [PortfolioStrategyConfig(**pcfg.__dict__) for pcfg in portfolio_configs_generic]
            else: 
                portfolio_configs_typed = [PortfolioStrategyConfig(**pcfg) for pcfg in portfolio_configs_generic]

        # Prepare prompts
        temp_node_attrs_pld3 = state.node_attrs.copy()
        temp_node_attrs_pld3["can_participate_this_round"] = eligible_final_voters_mask
        temp_state_pld_phase3 = state.replace(node_attrs=temp_node_attrs_pld3)

        # Get delegate declarations from cache for reminder prompts
        pld_delegate_declarations_cache = state.global_attrs.get("pld_delegate_declarations_cache_REF", {})
        pld_phase3_prompt_args = {"pld_current_phase_prompt_type": "pld_final_power_holder_vote"}
        
        for agent_idx in range(num_agents):
            if is_delegate_role[agent_idx] and eligible_final_voters_mask[agent_idx]:
                declared_votes_indices = pld_delegate_declarations_cache.get(agent_idx, [])
                if declared_votes_indices:
                    pld_phase3_prompt_args[f"pld_prior_declaration_agent_{agent_idx}"] = f"You previously declared support for portfolios: {declared_votes_indices}. Your final vote should be consistent unless you have a strong reason to change."
                else:
                    pld_phase3_prompt_args[f"pld_prior_declaration_agent_{agent_idx}"] = "You previously declared no portfolio approvals. Your final vote should be consistent unless you have a strong reason to change."

        agent_prompt_tasks_pld3, active_power_holders_pld3 = _get_llm_prompts_for_round(
            temp_state_pld_phase3, stable_sim_config,
            is_prd_election_voting_round=False, portfolio_configs=portfolio_configs_typed,
            candidate_original_indices=jnp.array([]), num_actual_candidates_for_election=0,
            pld_phase_specific_args=pld_phase3_prompt_args
        )
        
        llm_responses_pld3 = _execute_llm_calls_parallel(
            llm_service, agent_prompt_tasks_pld3, active_power_holders_pld3, round_num,
            getattr(stable_sim_config, 'max_llm_workers', 8)
        )

        # Parse Phase 3 responses
        current_portfolio_votes_for_phase3 = jnp.zeros((num_agents, len(portfolio_configs_typed) if portfolio_configs_typed else 0), dtype=jnp.int32)

        updated_portfolio_votes, _, _, decision_summary_pld3, round_anomalies_pld3 = _parse_llm_responses_and_prepare_updates(
            temp_state_pld_phase3, stable_sim_config, llm_responses_pld3, agent_prompt_tasks_pld3,
            is_prd_election_voting_round=False, portfolio_configs=portfolio_configs_typed, num_actual_candidates_for_election=0,
            new_agent_portfolio_votes=current_portfolio_votes_for_phase3,
            new_delegation_target=state.node_attrs.get("delegation_target"), 
            new_agent_election_candidate_votes=state.node_attrs.get("agent_votes_for_election_candidates")
        )
        
        if decision_summary_pld3 and (round_num < 3 or round_num % 20 == 0):
            print(f"[PLD_PHASE_3_VOTE] R{round_num} Decisions: {' | '.join(decision_summary_pld3)}")

        # FIXED: Smart enforcement - only override if delegate's Phase 3 response was invalid/empty
        for agent_idx in range(num_agents):
            if is_delegate_role[agent_idx] and eligible_final_voters_mask[agent_idx]:
                phase3_votes = updated_portfolio_votes[agent_idx, :]
                phase3_has_votes = jnp.sum(phase3_votes) > 0
                
                # Check if this delegate had a parse failure in Phase 3
                delegate_parse_failed = any(
                    f"A{agent_idx}" in log and "ParseFAIL" in log 
                    for log in decision_summary_pld3
                )
                
                # Only enforce Phase 1 declaration if Phase 3 failed or was empty
                if delegate_parse_failed or not phase3_has_votes:
                    declared_vote_indices = pld_delegate_declarations_cache.get(agent_idx, [])
                    if declared_vote_indices:
                        declared_vote_array = jnp.zeros(len(portfolio_configs_typed), dtype=jnp.int32)
                        declared_vote_array = declared_vote_array.at[jnp.array(declared_vote_indices, dtype=jnp.int32)].set(1)
                        updated_portfolio_votes = updated_portfolio_votes.at[agent_idx, :].set(declared_vote_array)
                        if round_num < 3 or round_num % 20 == 0:
                            print(f"[PLD_PHASE_3_ENFORCE] R{round_num} A{agent_idx}: Enforced Phase 1 declaration {declared_vote_indices} due to Phase 3 failure/empty")
                    else:
                        if round_num < 3 or round_num % 20 == 0:
                            print(f"[PLD_PHASE_3_ENFORCE] R{round_num} A{agent_idx}: No Phase 1 declaration to enforce, keeping Phase 3 result")
                else:
                    if round_num < 3 or round_num % 20 == 0:
                        print(f"[PLD_PHASE_3_KEEP] R{round_num} A{agent_idx}: Keeping valid Phase 3 response: {jnp.where(phase3_votes == 1)[0].tolist()}")

        final_node_attrs = state.node_attrs.copy()
        final_node_attrs["agent_portfolio_votes"] = updated_portfolio_votes
        
        final_global_attrs = state.global_attrs.copy()
        final_global_attrs["portfolio_decision_active_this_round"] = True
        final_global_attrs["round_anomaly_log"] = round_anomalies_pld3  # Store anomalies

        return state.replace(node_attrs=final_node_attrs, global_attrs=final_global_attrs)
    return transform

def create_prd_reps_vote_after_election_transform(
    llm_service: Optional[LLMService],
    stable_sim_config: StablePortfolioDemocracyConfig
) -> Transform:
    """
    A specialized transform that runs after a PRD election.
    It prompts ONLY the newly elected representatives for their portfolio votes.
    """
    def transform(state: GraphState) -> GraphState:
        round_num = state.global_attrs.get("round_num", 0)
        num_agents = state.num_nodes

        # This transform only acts if it's PRD and portfolio decisions were deferred due to an election.
        if not (stable_sim_config.mechanism == "PRD" and \
                state.global_attrs.get("portfolio_decision_active_this_round") == False):
            return state # Do nothing, pass state through

        print(f"[PRD_POST_ELECTION_VOTE] R{round_num}: Activating portfolio vote for newly elected representatives.")

        # Identify newly elected representatives
        elected_representatives_mask = state.node_attrs.get("is_elected_representative", jnp.zeros(num_agents, dtype=jnp.bool_))
        elected_agent_indices = jnp.where(elected_representatives_mask)[0].tolist()

        if not elected_agent_indices:
            print(f"[PRD_POST_ELECTION_VOTE] R{round_num}: No elected representatives found. Skipping portfolio vote.")
            # Ensure portfolio resolution can proceed, even if neutrally.
            new_global_attrs_no_reps = state.global_attrs.copy()
            new_global_attrs_no_reps["portfolio_decision_active_this_round"] = True
            return state.replace(global_attrs=new_global_attrs_no_reps)

        # --- Prepare for LLM calls, similar to the main decision transform but targeted ---
        portfolio_configs_generic = state.global_attrs.get("portfolio_configs", [])
        portfolio_configs_typed: List[PortfolioStrategyConfig] = []
        if portfolio_configs_generic:
            first_p_item = portfolio_configs_generic[0]
            if isinstance(first_p_item, PortfolioStrategyConfig): 
                portfolio_configs_typed = portfolio_configs_generic
            elif hasattr(first_p_item, '__dict__'): 
                portfolio_configs_typed = [PortfolioStrategyConfig(**pcfg.__dict__) for pcfg in portfolio_configs_generic]
            else: 
                portfolio_configs_typed = [PortfolioStrategyConfig(**pcfg) for pcfg in portfolio_configs_generic]

        # Create a temporary state or override for participation for this specific sub-prompting
        temp_can_participate_this_round = jnp.zeros(num_agents, dtype=jnp.bool_)
        if elected_agent_indices: # Ensure elected_agent_indices is not empty before trying to use it for indexing
            # Only elected reps who can generally participate should vote.
            general_participation_mask = state.node_attrs["can_participate_this_round"]
            actual_participating_elected_reps = elected_representatives_mask & general_participation_mask
            temp_can_participate_this_round = temp_can_participate_this_round.at[jnp.where(actual_participating_elected_reps)[0]].set(True)

        temp_state_for_prompting = state.replace(
            node_attrs={**state.node_attrs, "can_participate_this_round": temp_can_participate_this_round}
        )

        agent_prompt_tasks, active_llm_agents_for_reps_vote = _get_llm_prompts_for_round(
            temp_state_for_prompting, stable_sim_config, 
            is_prd_election_voting_round=False, # This is now a portfolio vote
            portfolio_configs=portfolio_configs_typed,
            candidate_original_indices=jnp.array([]), # Not relevant for this step
            num_actual_candidates_for_election=0      # Not relevant for this step
        )
        
        llm_responses_reps = _execute_llm_calls_parallel(
            llm_service, agent_prompt_tasks, active_llm_agents_for_reps_vote, round_num,
            getattr(stable_sim_config, 'max_llm_workers', 8)
        )

        # Parse responses - this will update agent_portfolio_votes
        num_portfolios_init = len(portfolio_configs_typed)
        current_portfolio_votes = state.node_attrs.get("agent_portfolio_votes", jnp.zeros((num_agents, num_portfolios_init if num_portfolios_init > 0 else 0), dtype=jnp.int32)).copy()
        
        updated_portfolio_votes, _, _, decision_summary_reps, round_anomalies_reps_vote = \
            _parse_llm_responses_and_prepare_updates(
                temp_state_for_prompting, stable_sim_config, llm_responses_reps, agent_prompt_tasks, # Use temp_state_for_prompting
                is_prd_election_voting_round=False, # It's a portfolio vote
                portfolio_configs=portfolio_configs_typed,
                num_actual_candidates_for_election=0, # Not relevant
                new_agent_portfolio_votes=current_portfolio_votes, 
                new_delegation_target=state.node_attrs.get("delegation_target", -jnp.ones(num_agents, dtype=jnp.int32)), # Not changed here
                new_agent_election_candidate_votes=state.node_attrs.get("agent_votes_for_election_candidates", jnp.zeros((num_agents,0), dtype=jnp.int32)) # Not changed here
            )
        if decision_summary_reps: print(f"[PRD_POST_ELECTION_VOTE] R{round_num} Decisions: {' | '.join(decision_summary_reps)}")

        final_node_attrs = state.node_attrs.copy()
        final_node_attrs["agent_portfolio_votes"] = updated_portfolio_votes
        
        final_global_attrs = state.global_attrs.copy()
        final_global_attrs["portfolio_decision_active_this_round"] = True # Now allow portfolio resolution
        
        # Append anomalies from this phase to existing ones from the election phase
        existing_anomalies = final_global_attrs.get("round_anomaly_log", [])
        final_global_attrs["round_anomaly_log"] = existing_anomalies + round_anomalies_reps_vote

        return state.replace(node_attrs=final_node_attrs, global_attrs=final_global_attrs)
    return transform


# --- Main Factory Function for Stable Democracy System ---
def create_stable_participation_mechanism_pipeline(
    stable_sim_config: StablePortfolioDemocracyConfig,
    llm_service: Optional[LLMService]
) -> Transform:
    
    inject_config_transform = lambda state: state.replace(
        global_attrs={**state.global_attrs, "stable_sim_config_REF": stable_sim_config}
    )
    housekeeping_transform = create_stable_housekeeping_transform()
    participation_constraint_transform = create_participation_constraint_transform()
    prediction_market_transform = create_stable_prediction_signal_transform()
    main_llm_decision_transform = create_llm_agent_decision_transform_stable(llm_service, stable_sim_config) # Renamed for clarity

    election_resolution_transform_prd = None
    if stable_sim_config.mechanism == "PRD":
        election_resolution_transform_prd = create_election_transform(
            candidate_flag_attr=getattr(stable_sim_config, 'prd_candidate_flag_attr', "is_delegate"),
            election_votes_attr="agent_votes_for_election_candidates"
        )

    # Portfolio path transforms
    delegation_update_transform_pld = create_delegation_transform() if stable_sim_config.mechanism == "PLD" else None
    power_flow_transform_pld = create_power_flow_transform() if stable_sim_config.mechanism == "PLD" else None
    portfolio_voting_transform = create_voting_transform(
        vote_aggregator=_portfolio_vote_aggregator_stable, # Use the correctly imported and implemented aggregator
        config={"mechanism_type": stable_sim_config.mechanism, "output_attr_name": "current_decision"}
    )
    apply_decision_to_resources_transform = create_resource_transform(
        resource_calculator=_portfolio_resource_calculator,
        config={"resource_attr_name": "current_total_resources", "history_attr_name": "resource_history"}
    )
    performance_update_transform = create_agent_performance_update_transform()

    # --- Assemble the pipeline ---
    pipeline_steps = [
        inject_config_transform,
        housekeeping_transform,
        participation_constraint_transform,
        prediction_market_transform,
        main_llm_decision_transform, # Handles PRD elections, PDD portfolio, PLD Phase 1 (Declaration) & PLD Phase 2 (Voter Delegation/Intent)
    ]

    # Election resolution for PRD (runs if it's an election round, after agent_decision made candidate votes)
    if stable_sim_config.mechanism == "PRD" and election_resolution_transform_prd:
        pipeline_steps.append(election_resolution_transform_prd)
        pipeline_steps.append(create_prd_election_debug_transform(stable_sim_config))  # Add debug
        # Post-election vote for PRD reps
        prd_reps_portfolio_vote_transform = create_prd_reps_vote_after_election_transform(llm_service, stable_sim_config)
        pipeline_steps.append(prd_reps_portfolio_vote_transform)

    # For PLD, specific sequence after main LLM decision (which handled Phase 1 & 2)
    if stable_sim_config.mechanism == "PLD":
        if delegation_update_transform_pld: pipeline_steps.append(delegation_update_transform_pld)
        if power_flow_transform_pld: pipeline_steps.append(power_flow_transform_pld)
        # PLD Phase 3: Final portfolio vote by power-holders
        pld_final_vote_transform = create_pld_final_power_holder_vote_transform(llm_service, stable_sim_config)
        pipeline_steps.append(pld_final_vote_transform)
    
    # --- Portfolio Resolution Steps ---
    # The `portfolio_decision_active_this_round` flag manages this.
    # - main_llm_decision_transform sets it False for exclusive PRD elections or for PLD.
    # - create_prd_reps_vote_after_election_transform sets it True after PRD reps vote.
    # - create_pld_final_power_holder_vote_transform sets it True after PLD Phase 3.

    def gate_portfolio_resolution(next_transform: Transform) -> Transform:
        def gated_transform(state: GraphState) -> GraphState:
            # Get the flag set by agent_decision_transform (or by election_resolution if it concludes the round action)
            make_portfolio_decision_this_round = state.global_attrs.get("portfolio_decision_active_this_round", True)
            
            if make_portfolio_decision_this_round:
                return next_transform(state)
            else:
                # Skip portfolio resolution, but ensure 'current_decision' is neutral if not set
                print(f"[Round {state.global_attrs.get('round_num',-1)}] Skipping portfolio resolution (e.g. PRD election-only round).")
                new_globals = state.global_attrs.copy()
                if "current_decision" not in new_globals : new_globals["current_decision"] = -1 # Neutral/no decision
                
                # Housekeeping should have set current_actual_crop_yields.
                # This is a fallback if, for some reason, it's missing when portfolio resolution is skipped.
                if "current_actual_crop_yields" not in new_globals : # Ensure this exists if resource calc runs
                    num_crops_val = len(stable_sim_config.crops) if stable_sim_config.crops else 0
                    if num_crops_val > 0:
                        print(f"[WARN] R{state.global_attrs.get('round_num',-1)}: current_actual_crop_yields was not set. Defaulting in gate.")
                        new_globals["current_actual_crop_yields"] = jnp.ones(num_crops_val, dtype=jnp.float32)
                    else:
                        new_globals["current_actual_crop_yields"] = jnp.array([], dtype=jnp.float32)
                return state.replace(global_attrs=new_globals)
        return gated_transform
    
    pipeline_steps.extend([
        gate_portfolio_resolution(portfolio_voting_transform),
        gate_portfolio_resolution(apply_decision_to_resources_transform),
        gate_portfolio_resolution(performance_update_transform) # Performance also depends on a portfolio choice
    ])
        
    return sequential(*pipeline_steps)

# Add this after the election resolution transform
def create_prd_election_debug_transform(stable_sim_config: StablePortfolioDemocracyConfig) -> Transform:
    def debug_prd_election(state: GraphState) -> GraphState:
        round_num = state.global_attrs.get("round_num", 0)
        elected_mask = state.node_attrs.get("is_elected_representative", jnp.zeros(state.num_nodes, dtype=jnp.bool_))
        is_adversarial = state.node_attrs.get("is_adversarial", jnp.zeros(state.num_nodes, dtype=jnp.bool_))
        
        elected_indices = jnp.where(elected_mask)[0]
        adversarial_elected = jnp.sum(elected_mask & is_adversarial)
        total_elected = jnp.sum(elected_mask)
        total_adversarial = jnp.sum(is_adversarial)
        
        print(f"[PRD_ELECTION_DEBUG] R{round_num}: {adversarial_elected}/{total_elected} adversarial reps elected")
        print(f"[PRD_ELECTION_DEBUG] R{round_num}: Total adversarial agents: {total_adversarial}/{state.num_nodes}")
        print(f"[PRD_ELECTION_DEBUG] R{round_num}: Elected agents: {elected_indices.tolist()}")
        
        # Check if adversarial agents are being strategic
        for i, agent_id in enumerate(elected_indices):
            is_adv = is_adversarial[agent_id]
            print(f"[PRD_ELECTION_DEBUG] R{round_num}: Agent {agent_id} elected, adversarial: {is_adv}")
        
        return state
    
    return debug_prd_election