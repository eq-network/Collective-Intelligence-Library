from typing import Literal, Dict, Any, Optional, Callable, List
import jax
import jax.numpy as jnp
import jax.random as jr
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from algebra.category import Transform, sequential
from algebra.graph import GraphState

# Enhanced imports
from environments.democracy.random.configuration import (
    PortfolioDemocracyConfig, 
    PromptConfig,
    PortfolioStrategyConfig,
    CropConfig,
    create_thesis_baseline_config
)
from environments.democracy.random.optimality_analysis import (
    OptimalityCalculator,
    PerformanceAnalyzer,
    calculate_optimality_for_state,
    generate_optimality_prompt_text
)
from services.llm import LLMService

# Import existing transforms (unchanged)
from transformations.bottom_up.prediction_market import create_prediction_market_transform
from environments.democracy.transforms.delegation import create_delegation_transform
from environments.democracy.transforms.power_flow import create_power_flow_transform
from environments.democracy.transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform
from environments.democracy.transforms.election import create_election_transform

# --- Helper Transforms and Calculators (largely unchanged from your version) ---

def create_start_of_round_housekeeping_transform() -> Transform:
    """Resets per-round agent states like tokens spent and updates round num."""
    def transform(state: GraphState) -> GraphState:
        new_node_attrs = dict(state.node_attrs)
        if "tokens_spent_current_round" in new_node_attrs:
            new_node_attrs["tokens_spent_current_round"] = jnp.zeros_like(new_node_attrs["tokens_spent_current_round"])
        
        new_global_attrs = dict(state.global_attrs)
        new_global_attrs["round_num"] = state.global_attrs.get("round_num", -1) + 1
        return state.replace(node_attrs=new_node_attrs, global_attrs=new_global_attrs)
    return transform


def _portfolio_vote_aggregator(state: GraphState, transform_config: Dict[str, Any]) -> jnp.ndarray:
    agent_votes = state.node_attrs["agent_portfolio_votes"] 
    mechanism_type = transform_config.get("mechanism_type", "direct")

    if mechanism_type == "direct": 
        aggregated_votes = jnp.sum(agent_votes, axis=0)
    elif mechanism_type == "representative": 
        is_delegate = state.node_attrs["is_delegate"]
        delegate_votes = agent_votes * is_delegate[:, jnp.newaxis]
        aggregated_votes = jnp.sum(delegate_votes, axis=0)
    elif mechanism_type == "liquid": 
        voting_power = state.node_attrs["voting_power"] 
        weighted_votes = agent_votes * voting_power[:, jnp.newaxis]
        aggregated_votes = jnp.sum(weighted_votes, axis=0)
    else:
        raise ValueError(f"Unknown mechanism_type for vote aggregation: {mechanism_type}")
    return aggregated_votes

def create_actual_yield_sampling_transform() -> Transform:
    def transform(state: GraphState) -> GraphState:
        key_val = state.global_attrs.get("round_num", 0) + state.global_attrs.get("simulation_seed", 0) + 1 
        key = jr.PRNGKey(key_val)
        
        crop_configs_generic = state.global_attrs["crop_configs"]
        # Simple conversion to CropConfig objects
        crop_configs = [
            CropConfig(**cc.__dict__) if hasattr(cc, '__dict__') else cc 
            for cc in crop_configs_generic
        ]

        if not crop_configs: # Handle empty crop_configs
            new_global_attrs = dict(state.global_attrs)
            new_global_attrs["current_actual_crop_yields"] = jnp.array([])
            return state.replace(global_attrs=new_global_attrs)

        true_expected_yields = state.global_attrs["current_true_expected_crop_yields"]
        print(f"DEBUG: True expected yields: {true_expected_yields}")
        
        actual_yields = []
        crop_keys = jr.split(key, len(crop_configs))

        for i, crop_cfg in enumerate(crop_configs):
            variance_beta = (crop_cfg.yield_beta_dist_alpha * crop_cfg.yield_beta_dist_beta) / \
                            ((crop_cfg.yield_beta_dist_alpha + crop_cfg.yield_beta_dist_beta)**2 * \
                             (crop_cfg.yield_beta_dist_alpha + crop_cfg.yield_beta_dist_beta + 1))
            sample_sigma = jnp.sqrt(variance_beta) * true_expected_yields[i] * 0.5 
            sampled_deviation = jr.normal(crop_keys[i]) * sample_sigma
            actual_yield = true_expected_yields[i] + sampled_deviation
            actual_yields.append(jnp.maximum(0.0, actual_yield)) 
        
        actual_yields_array = jnp.array(actual_yields)
        print(f"DEBUG: Actual sampled yields: {actual_yields_array}")

        new_global_attrs = dict(state.global_attrs)
        new_global_attrs["current_actual_crop_yields"] = actual_yields_array
        return state.replace(global_attrs=new_global_attrs)
    return transform

def log_decision_transform(state: GraphState) -> GraphState:
    round_num = state.global_attrs.get("round_num", -1)
    mech = state.global_attrs.get("stable_sim_config_REF").mechanism # Get mechanism
    decision = state.global_attrs.get("current_decision", -99)
    vote_dist = state.global_attrs.get("vote_distribution", "N/A")
    print(f"[PRE_RESOURCE_CALC] R{round_num} ({mech}): current_decision = {decision}, vote_distribution = {vote_dist}")
    return state


def _portfolio_resource_calculator(state: GraphState, transform_config: Dict[str, Any]) -> float:
    chosen_portfolio_idx = state.global_attrs.get("current_decision")
    print(f"DEBUG: Portfolio decision index: {chosen_portfolio_idx}")
    
    if chosen_portfolio_idx is None:
        print("DEBUG: No portfolio decision found")
        return 1.0

    portfolio_configs_generic = state.global_attrs["portfolio_configs"]
    portfolio_configs = [
        PortfolioStrategyConfig(**ps.__dict__) if hasattr(ps, '__dict__') else ps
        for ps in portfolio_configs_generic
    ]
    actual_crop_yields = state.global_attrs["current_actual_crop_yields"]
    print(f"DEBUG: Actual crop yields: {actual_crop_yields}")
    
    if not (0 <= chosen_portfolio_idx < len(portfolio_configs)):
        print(f"DEBUG: Invalid portfolio index: {chosen_portfolio_idx}")
        return 1.0

    selected_portfolio = portfolio_configs[chosen_portfolio_idx]
    portfolio_weights = jnp.array(selected_portfolio.weights)
    
    if portfolio_weights.shape[0] != actual_crop_yields.shape[0]:
        print(f"DEBUG: Weight/Yield shape mismatch: {portfolio_weights.shape} vs {actual_crop_yields.shape}")
        return 1.0
        
    portfolio_return = jnp.sum(portfolio_weights * actual_crop_yields)
    print(f"DEBUG: Portfolio {chosen_portfolio_idx} ({selected_portfolio.name}) return: {portfolio_return}")
    
    return float(portfolio_return)

def create_llm_agent_decision_transform(
    llm_service: Optional[LLMService],
    mechanism: Literal["PDD", "PRD", "PLD"],
    sim_config: PortfolioDemocracyConfig
) -> Transform:
    """
    LLM agent decision transform with ENHANCED debug output for responses.
    
    DEBUG STRATEGY:
    - Show LLM response snippets for decision transparency
    - Track agent decision patterns
    - Provide round-level decision summaries
    - Remove excessive technical details
    """
    def transform(state: GraphState) -> GraphState:
        num_agents = state.num_nodes
        portfolio_configs = state.global_attrs["portfolio_configs"]
        num_portfolios = len(portfolio_configs)
        
        # Get agent-specific prediction signals
        agent_specific_signals = state.global_attrs.get("agent_specific_prediction_signals", {})
        # uniform_signals (old "prediction_market_crop_signals") is no longer the primary source if agent_specific_signals is populated.
        
        # Initialize outputs
        new_agent_portfolio_votes = state.node_attrs.get("agent_portfolio_votes", 
            jnp.zeros((num_agents, num_portfolios), dtype=jnp.int32)).copy()
        new_delegation_target = state.node_attrs.get("delegation_target", 
            -jnp.ones(num_agents, dtype=jnp.int32)).copy()
        new_tokens_spent = state.node_attrs.get("tokens_spent_current_round", 
            jnp.zeros(num_agents, dtype=jnp.int32)).copy()

        # Round-level decision tracking
        round_num = state.global_attrs.get("round_num", 0)
        decision_summary = []
        
        # --- Prepare data for all agents first ---
        agent_prompt_tasks = []

        for i in range(num_agents): # First loop to prepare tasks
            agent_key_val = state.global_attrs.get("round_num", 0) + state.global_attrs.get("simulation_seed", 0) + i + 1000
            agent_key = jr.PRNGKey(agent_key_val)

            is_adversarial = bool(state.node_attrs["is_adversarial"][i])
            is_delegate_role = bool(state.node_attrs["is_delegate"][i])
            
            # Get cognitive resources
            if is_delegate_role:
                cognitive_resources = sim_config.cognitive_resource_settings.cognitive_resources_delegate
            else:
                cognitive_resources = sim_config.cognitive_resource_settings.cognitive_resources_voter
            
            # Determine active participation
            is_active_voter_for_round = True
            if mechanism == "PRD":
                if not state.node_attrs["is_elected_representative"][i]:
                    is_active_voter_for_round = False
            
            if not is_active_voter_for_round:
                continue

            # Use agent-specific prediction signals.
            # These are now expected to be generated for each agent.
            if i in agent_specific_signals:
                agent_pm_signals = agent_specific_signals[i]
            else:
                # Fallback if somehow agent_specific_signals is not populated for this agent
                print(f"[WARNING] Agent {i} missing from agent_specific_prediction_signals. Using default (potentially zeros).")
                agent_pm_signals = jnp.zeros(len(sim_config.crops)) # Or handle error more gracefully

            # Generate portfolio expected yields string
            portfolio_expected_yields = []
            for p_cfg in portfolio_configs:
                p_weights = jnp.array(p_cfg.weights)
                expected_yield = jnp.sum(p_weights * agent_pm_signals)
                portfolio_expected_yields.append(f"{p_cfg.name} (Predicted Yield: {expected_yield:.2f}x)")
            
            portfolio_options_str = "\n".join([f"{i}: {desc}" for i, desc in enumerate(portfolio_expected_yields)])

            # Prepare delegation targets info
            delegate_targets_info = None
            if mechanism == "PLD":
                delegation_targets = []
                for k in range(num_agents):
                    if k != i and state.node_attrs["is_delegate"][k]:
                        delegation_targets.append(f"  Agent {k} (Designated Delegate)")
                if delegation_targets:
                    delegate_targets_info = "Potential Delegation Targets:\n" + "\n".join(delegation_targets)

            # Generate prompt
            optimality_analysis_str = True
            if sim_config.include_optimality_analysis:
                # Ensure OptimalityCalculator is imported and available
                optimality_calculator = OptimalityCalculator() 
                try:
                    current_signals = agent_pm_signals # Use agent-specific signals if available
                    # Ensure portfolio_configs is a list of PortfolioStrategyConfig
                    # It should be from state.global_attrs["portfolio_configs"]
                    optimality_result = optimality_calculator.calculate_portfolio_optimality(
                        prediction_signals=current_signals,
                        portfolio_configs=portfolio_configs 
                    )
                    optimality_analysis_str = generate_optimality_prompt_text(optimality_result)
                except Exception as e_opt:
                    print(f"[OPTIMALITY_ERROR] Agent {i}, Round {round_num}: {e_opt}")
                    optimality_analysis_str = "Optimality analysis currently unavailable."

            # Generate performance history string for delegation decisions
            performance_history_str = True
            if sim_config.use_redteam_prompts and mechanism == "PLD":
                history_lines = ["DELEGATE PERFORMANCE HISTORY (recent performance score 0.0-1.0, 1.0 is best):"]
                
                # These attributes are now calculated and updated by create_agent_performance_update_transform
                cumulative_scores = state.node_attrs.get("cumulative_performance_score", jnp.full(num_agents, 0.0)) 
                decisions_made_count = state.node_attrs.get("num_decisions_made_history", jnp.zeros(num_agents, dtype=jnp.int32))

                delegates_found_for_history = False
                for k in range(num_agents):
                    # Agent k is a delegate, is not the current agent i, and is a potential target
                    if state.node_attrs["is_delegate"][k] and k != i: 
                        delegates_found_for_history = True
                        score = float(cumulative_scores[k])
                        count = int(decisions_made_count[k])
                        history_lines.append(
                            f"  Agent {k}: Performance Score: {score:.2f}/1.0 (based on {count} decisions)"
                        )
                if delegates_found_for_history:
                    performance_history_str = "\n".join(history_lines)
                else:
                    performance_history_str = "No delegate performance history available for other agents."

            # Directly use the unified generate_prompt method
            prompt_result = sim_config.prompt_settings.generate_prompt(
                    agent_id=i,
                    round_num=round_num,
                    is_delegate=is_delegate_role,
                    is_adversarial=is_adversarial,
                    cognitive_resources=cognitive_resources, # This is the 0-100 score
                    mechanism=mechanism,
                    portfolio_options_str=portfolio_options_str,
                    delegate_targets_str=delegate_targets_info,
                    performance_history_str=performance_history_str,
                    optimality_analysis=optimality_analysis_str,
                    include_decision_framework=True # Or control this via sim_config if needed
            )

            prompt = prompt_result["prompt"]
            max_tokens = prompt_result["max_tokens"]

            agent_prompt_tasks.append({
                "agent_id": i,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "is_adversarial": is_adversarial,
                "is_delegate_role": is_delegate_role,
                "is_active_voter_for_round": is_active_voter_for_round,
                "mechanism": mechanism # Store mechanism for parsing logic
            })

        # --- Execute LLM calls in parallel (if llm_service exists) ---
        llm_responses = {} # Store {agent_id: llm_response_text}

        if llm_service:
            # Limit concurrent LLM calls to avoid overwhelming APIs or local resources
            # This number might need tuning based on API limits and num_agents
            max_llm_workers = min(num_agents, 8) # Example: up to 8 concurrent LLM calls
            with ThreadPoolExecutor(max_workers=max_llm_workers) as executor:
                future_to_agent_id = {
                    executor.submit(llm_service.generate, task["prompt"], task["max_tokens"]): task["agent_id"]
                    for task in agent_prompt_tasks if task["is_active_voter_for_round"]
                }
                for future in as_completed(future_to_agent_id):
                    agent_id = future_to_agent_id[future]
                    try:
                        llm_responses[agent_id] = future.result()
                    except Exception as e:
                        print(f"[R{round_num:02d}] LLM error for Agent {agent_id} (parallel): {e}")
                        llm_responses[agent_id] = "" # Empty response on error

        # --- Process responses and apply decisions ---
        for task_data in agent_prompt_tasks:
            i = task_data["agent_id"]
            if not task_data["is_active_voter_for_round"]:
                continue

            llm_response_text = llm_responses.get(i, "") # Get from parallel execution or empty if no LLM/error
            
            chosen_portfolio_indices_to_approve = []
            delegation_choice = -1
            action_cost = 0
            agent_action_this_round = False
            is_adversarial = task_data["is_adversarial"]
            is_delegate_role = task_data["is_delegate_role"]
            current_mechanism = task_data["mechanism"] # Use stored mechanism

            # ENHANCED DEBUG: Show meaningful LLM response snippets
            # response_snippet = llm_response_text[:100] + "..." if len(llm_response_text) > 100 else llm_response_text
            agent_role_str = "Delegate" if is_delegate_role else "Voter"
            agent_type_str = "Adversarial" if is_adversarial else "Aligned"
            
            if llm_response_text:
                if current_mechanism == "PLD":
                    action_match = re.search(r"Action:\s*(\w+)", llm_response_text, re.IGNORECASE)
                    if action_match and action_match.group(1).upper() == "DELEGATE":
                        target_match = re.search(r"AgentID:\s*(\d+)", llm_response_text, re.IGNORECASE)
                        if target_match:
                            # ... (existing delegation logic) ...
                            agent_action_this_round = True
                            # ...
                        else: # Action DELEGATE but no AgentID
                            print(f"[DEBUG_PARSE_WARN] R{round_num} A{i} (PLD): Action DELEGATE but no AgentID found. LLM Response: '{llm_response_text[:200]}...'")
                            decision_summary.append(f"A{i}({agent_type_str[0]}{agent_role_str[0]}) → DELEGATE FAIL (no target)")


                # This block handles VOTE for PLD, or any action for PDD/PRD
                if not agent_action_this_round: # If not PLD delegation, try to parse votes
                    votes_match = re.search(r"Votes:\s*\[([^\]]*)\]", llm_response_text, re.IGNORECASE)
                    if votes_match:
                        try:
                            vote_str_list = votes_match.group(1).split(',')
                            parsed_votes = [int(v.strip()) for v in vote_str_list if v.strip().isdigit()] # make more robust
                            if not parsed_votes and vote_str_list and any(v.strip() for v in vote_str_list): # e.g. Votes: [blah]
                                print(f"[DEBUG_PARSE_WARN] R{round_num} A{i} ({current_mechanism}): 'Votes: [...]' found but content not all 0/1. Content: '{vote_str_list}'. LLM Response: '{llm_response_text[:200]}...'")
                                # Defaults to empty chosen_portfolio_indices_to_approve

                            if len(parsed_votes) == num_portfolios:
                                chosen_portfolio_indices_to_approve = [idx for idx, val in enumerate(parsed_votes) if val == 1]
                                agent_action_this_round = True # Valid vote format processed
                                if chosen_portfolio_indices_to_approve:
                                    portfolio_names = [portfolio_configs[idx].name for idx in chosen_portfolio_indices_to_approve]
                                    decision_summary.append(f"A{i}({agent_type_str[0]}{agent_role_str[0]}) → VOTE {portfolio_names}")
                                else: # e.g. Votes: [0,0,0]
                                    decision_summary.append(f"A{i}({agent_type_str[0]}{agent_role_str[0]}) → VOTE FOR NO PORTFOLIOS")
                            else: # Parsed votes, but wrong number
                                print(f"[DEBUG_PARSE_WARN] R{round_num} A{i} ({current_mechanism}): Parsed {len(parsed_votes)} votes, expected {num_portfolios}. LLM Response: '{llm_response_text[:200]}...'")
                                # chosen_portfolio_indices_to_approve remains empty, agent_action_this_round still False
                        except ValueError:
                            print(f"[DEBUG_PARSE_ERROR] R{round_num} A{i} ({current_mechanism}): ValueError parsing votes. LLM Response: '{llm_response_text[:200]}...'")
                            # chosen_portfolio_indices_to_approve remains empty, agent_action_this_round still False
                    else: # No "Votes: [...]" match
                        # This is the critical path for PDD/PRD if LLM is confused
                        if current_mechanism != "PLD" or (current_mechanism == "PLD" and (not action_match or action_match.group(1).upper() == "VOTE")):
                             print(f"[DEBUG_PARSE_FAIL] R{round_num} A{i} ({current_mechanism}): No 'Votes: [...]' found. LLM Response: '{llm_response_text[:200]}...'")
                             decision_summary.append(f"A{i}({agent_type_str[0]}{agent_role_str[0]}) → VOTE PARSE FAIL")
            else: # No LLM response text
                 print(f"[DEBUG_NO_RESPONSE] R{round_num} A{i} ({current_mechanism}): No LLM response text received.")
                 decision_summary.append(f"A{i}({agent_type_str[0]}{agent_role_str[0]}) → NO LLM RESPONSE")

            # Fallback / default action if no valid action was parsed
            if not agent_action_this_round:
                print(f"[DEBUG_FALLBACK] R{round_num} A{i} ({current_mechanism}): No valid action parsed. Defaulting to NO VOTE / NO ACTION.")
                agent_action_this_round = True # Count as an action (of doing nothing useful)
                # For PDD/PRD, or PLD vote fail, this means chosen_portfolio_indices_to_approve is empty
                # new_agent_portfolio_votes.at[i].set(jnp.zeros(num_portfolios, dtype=jnp.int32)) will handle this.
                # For PLD delegation fail, new_delegation_target.at[i].set(-1) will handle this.
                if current_mechanism == "PLD": # If it was PLD and they didn't make a valid DELEGATE or VOTE
                    new_delegation_target = new_delegation_target.at[i].set(-1) # Ensure no prior delegation target sticks
                    new_agent_portfolio_votes = new_agent_portfolio_votes.at[i].set(jnp.zeros(num_portfolios, dtype=jnp.int32))


            # Apply decisions (this part should largely remain the same)
            if agent_action_this_round: # This flag now means "an action, even if it's a parsed failure/default, has been accounted for"
                new_tokens_spent = new_tokens_spent.at[i].add(action_cost)

                if delegation_choice != -1 and current_mechanism == "PLD": # Successfully chose to delegate
                    new_delegation_target = new_delegation_target.at[i].set(delegation_choice)
                    new_agent_portfolio_votes = new_agent_portfolio_votes.at[i].set(jnp.zeros(num_portfolios, dtype=jnp.int32))
                else: # Voted (or attempted to vote / defaulted to no vote)
                    current_agent_votes = jnp.zeros(num_portfolios, dtype=jnp.int32)
                    if chosen_portfolio_indices_to_approve: # This will be empty if parsing failed or voted all zeros
                        current_agent_votes = current_agent_votes.at[jnp.array(chosen_portfolio_indices_to_approve, dtype=jnp.int32)].set(1)
                    new_agent_portfolio_votes = new_agent_portfolio_votes.at[i].set(current_agent_votes)
                    if mechanism == "PLD": # If PLD and voted (or failed to vote/delegate), ensure delegation target is -1
                        new_delegation_target = new_delegation_target.at[i].set(-1)

        # ENHANCED DEBUG: Round-level decision summary
        if decision_summary and len(decision_summary) <= 5:  # Show max 5 for readability
            print(f"[R{round_num:02d}] Agent Decisions: {' | '.join(decision_summary[:5])}")
        elif decision_summary:
            print(f"[R{round_num:02d}] Agent Decisions: {len(decision_summary)} agents made decisions")

        # Update node attributes
        new_node_attrs = dict(state.node_attrs)
        new_node_attrs["agent_portfolio_votes"] = new_agent_portfolio_votes
        if mechanism == "PLD":
            new_node_attrs["delegation_target"] = new_delegation_target
        
        return state.replace(node_attrs=new_node_attrs)
    
    return transform

def create_agent_performance_update_transform() -> Transform:
    """
    Calculates and updates each agent's historical performance score based on
    their portfolio choices in the current round, relative to the optimal
    choice given their own prediction signals.
    """
    def transform(state: GraphState) -> GraphState:
        num_agents = state.num_nodes
        portfolio_configs = state.global_attrs["portfolio_configs"]
        
        agent_specific_signals = state.global_attrs.get("agent_specific_prediction_signals", {})
        uniform_signals = state.global_attrs.get("prediction_market_crop_signals") # Fallback
        
        agent_portfolio_votes = state.node_attrs["agent_portfolio_votes"] # Shape: (num_agents, num_portfolios)
        
        # Get current historical performance data
        cumulative_scores = state.node_attrs["cumulative_performance_score"].copy()
        num_decisions = state.node_attrs["num_decisions_made_history"].copy()

        for i in range(num_agents):
            # Check if agent i made a portfolio choice (i.e., has any '1' in their vote vector)
            # This also implicitly checks if they were an active voter for the round.
            # In PLD, if an agent delegates, their agent_portfolio_votes for that round will be all zeros.
            if jnp.sum(agent_portfolio_votes[i]) == 0:
                continue # Agent did not vote for a portfolio (e.g., delegated or inactive)

            # Get the agent's prediction signals for this round
            if i in agent_specific_signals:
                current_agent_signals = agent_specific_signals[i]
            elif uniform_signals is not None:
                current_agent_signals = uniform_signals
            else:
                # Should not happen if prediction market ran, but as a safeguard:
                print(f"Warning: No prediction signals found for agent {i} to calculate performance.")
                continue

            # Calculate expected returns for all portfolios based on THIS agent's signals
            expected_returns_for_agent = []
            for p_cfg in portfolio_configs:
                p_weights = jnp.array(p_cfg.weights)
                expected_return = jnp.sum(p_weights * current_agent_signals)
                expected_returns_for_agent.append(float(expected_return))
            
            if not expected_returns_for_agent:
                continue

            min_r = min(expected_returns_for_agent)
            max_r = max(expected_returns_for_agent)

            # Determine the expected return of the agent's chosen portfolio(s)
            chosen_indices = jnp.where(agent_portfolio_votes[i] == 1)[0]
            if len(chosen_indices) == 0: # Should be caught by the sum check above, but for safety
                continue

            chosen_portfolio_returns = [expected_returns_for_agent[idx] for idx in chosen_indices]
            avg_chosen_return = sum(chosen_portfolio_returns) / len(chosen_portfolio_returns)

            # Calculate current round's performance score (0-1)
            current_round_score = 0.5 # Default for edge cases like max_r == min_r
            if max_r > min_r:
                current_round_score = (avg_chosen_return - min_r) / (max_r - min_r)
            elif max_r == min_r : # All options were perceived as equal by the agent
                 current_round_score = 1.0 # Agent made the best possible choice given their info

            current_round_score = jnp.clip(current_round_score, 0.0, 1.0) # Ensure score is 0-1

            # Update running average for historical performance
            old_total_score = cumulative_scores[i] * num_decisions[i]
            new_num_decisions = num_decisions[i] + 1
            cumulative_scores = cumulative_scores.at[i].set((old_total_score + current_round_score) / new_num_decisions)
            num_decisions = num_decisions.at[i].set(new_num_decisions)

        new_node_attrs = dict(state.node_attrs)
        new_node_attrs["cumulative_performance_score"] = cumulative_scores
        new_node_attrs["num_decisions_made_history"] = num_decisions
        return state.replace(node_attrs=new_node_attrs)
    return transform

# --- Main Factory Function (largely unchanged from your version, ensure create_llm_agent_decision_transform is called) ---

def create_portfolio_mechanism_pipeline(
    mechanism: Literal["PDD", "PRD", "PLD"],
    llm_service: Optional[LLMService], # Added LLMService here
    sim_config: PortfolioDemocracyConfig 
) -> Transform:
    
    housekeeping_transform = create_start_of_round_housekeeping_transform()

    # Initialize election_transform_prd to None so it's always bound
    election_transform_prd = None 

    # --- New Agent-Specific Prediction Market Signal Generator ---
    def _agent_specific_prediction_market_signal_generator(state: GraphState, generator_config: Dict[str, Any]) -> Dict[int, jnp.ndarray]:
        """
        Generates prediction market signals for each agent, scaled by their cognitive resources.
        """
        # generator_config is passed by create_prediction_market_transform but not used here.
        agent_perceived_signals = {}
        true_expected_yields = state.global_attrs["current_true_expected_crop_yields"]
        base_noise_sigma = sim_config.market_settings.prediction_noise_sigma # Global base noise
        
        num_agents = state.num_nodes
        # Ensure "cognitive_resources" attribute exists and is correctly populated during initialization
        agent_cognitive_resources = state.node_attrs.get("cognitive_resources", jnp.full(num_agents, 50.0)) # Default to 50 if missing

        base_key_val = state.global_attrs.get("round_num", 0) + state.global_attrs.get("simulation_seed", 0)
        
        print(f"PREDICTION_MARKET_DEBUG (Agent-Specific): Round {state.global_attrs.get('round_num', 0)}")
        print(f"  Base Noise Sigma: {base_noise_sigma}")
        print(f"  True Expected Crop Yields: {true_expected_yields}")

        for agent_id in range(num_agents):
            key_agent_noise = jr.PRNGKey(base_key_val + agent_id + 500) # Unique key per agent per round
            
            cog_res = jnp.clip(agent_cognitive_resources[agent_id], 0, 100) # Ensure CR is within 0-100
            
            # Noise multiplier: 1.0 for CR=0 (full base_noise_sigma), 0.0 for CR=100 (no noise)
            # CR=20 -> (100-20)/100 = 0.8 multiplier
            # CR=80 -> (100-80)/100 = 0.2 multiplier
            noise_multiplier = (100.0 - cog_res) / 100.0
            
            agent_specific_sigma = base_noise_sigma * noise_multiplier
            
            noise = jr.normal(key_agent_noise, shape=true_expected_yields.shape) * agent_specific_sigma
            agent_perceived_signals[agent_id] = true_expected_yields + noise

            if agent_id == 0 and state.global_attrs.get('round_num', 0) < 1: # Log for one agent in early rounds
                print(f"  Agent {agent_id} (CR: {cog_res:.0f}): Noise Multiplier: {noise_multiplier:.2f}, Agent Sigma: {agent_specific_sigma:.4f}, Signals: {agent_perceived_signals[agent_id]}")

        return agent_perceived_signals
    # --- End New Generator ---
    def log_decision_transform(state: GraphState) -> GraphState:
        round_num = state.global_attrs.get("round_num", -1)
        mech = state.global_attrs.get("stable_sim_config_REF").mechanism # Get mechanism
        decision = state.global_attrs.get("current_decision", -99)
        vote_dist = state.global_attrs.get("vote_distribution", "N/A")
        print(f"[PRE_RESOURCE_CALC] R{round_num} ({mech}): current_decision = {decision}, vote_distribution = {vote_dist}")
        return state

    prediction_market_transform = create_prediction_market_transform(
        prediction_generator=_agent_specific_prediction_market_signal_generator, # Use the new generator
        config={"output_attr_name": "agent_specific_prediction_signals"} # Store output here
    )
    
    # THIS IS THE KEY CHANGE: Pass llm_service and sim_config
    agent_decision_transform = create_llm_agent_decision_transform(
        llm_service, mechanism, sim_config
    )

    delegation_related_transforms = []
    voting_config_key = "direct" 

    if mechanism == "PLD":
        voting_config_key = "liquid"
        delegation_update = create_delegation_transform() 
        power_flow = create_power_flow_transform()      
        delegation_related_transforms = [delegation_update, power_flow]
    elif mechanism == "PRD":
        voting_config_key = "representative"
        election_transform_prd = create_election_transform(election_logic="highest_cog_resource") # or "random_approval"

    voting_transform = create_voting_transform(
        vote_aggregator=_portfolio_vote_aggregator,
        config={"mechanism_type": voting_config_key, "output_attr_name": "current_decision"}
    )

    actual_yield_transform = create_actual_yield_sampling_transform() 

    apply_decision_to_resources_transform = create_resource_transform(
        resource_calculator=_portfolio_resource_calculator,
        config={
            "resource_attr_name": "current_total_resources", 
            "history_attr_name": "resource_history"
            } 
    )
    
    performance_update_transform = create_agent_performance_update_transform()

    pipeline_steps = []
    if election_transform_prd: # If PRD, election logic runs first (or after housekeeping)
        pipeline_steps.append(housekeeping_transform)
        pipeline_steps.append(election_transform_prd)
    else:
        pipeline_steps.append(housekeeping_transform)


    pipeline_steps.extend([
        prediction_market_transform, 
        agent_decision_transform,    
    ])
    pipeline_steps.extend(delegation_related_transforms) 
    pipeline_steps.extend([
        voting_transform,
        log_decision_transform,            
        actual_yield_transform,      
        apply_decision_to_resources_transform,
        performance_update_transform # Add performance update at the end of the round
    ])
    
    return sequential(*pipeline_steps)

def run_enhanced_mechanism_comparison():
    """
    Example function showing how to use the enhanced system for mechanism comparison.
    
    This demonstrates the complete integration of red team prompting and optimality analysis.
    """
    
    print("🔴 Enhanced Mechanism Comparison with Red Team Prompting and Optimality Analysis")
    print("=" * 80)
    
    # Create enhanced configurations
    mechanisms = ["PDD", "PLD", "PRD"]
    results = {}
    
    for mechanism in mechanisms:
        print(f"\n🧪 Testing {mechanism} with enhanced prompting...")
        
        # Create enhanced configuration
        config = create_thesis_baseline_config(
            mechanism=mechanism,
            adversarial_proportion_total=0.3,
            seed=12345,
        )
        
        print(f"  Configuration: {config.num_agents} agents, {config.num_delegates} delegates")
        print(f"  Optimality analysis: {config.include_optimality_analysis}")
        
        # Initialize simulation state
        key = jr.PRNGKey(config.seed)
        from environments.random.initialization import initialize_portfolio_democracy_graph_state
        initial_state = initialize_portfolio_democracy_graph_state(key, config)
        
        # Calculate baseline optimality
        try:
            optimality = calculate_optimality_for_state(initial_state)
            print(f"  Optimal portfolio: {optimality.optimal_choice} (return: {optimality.expected_returns[optimality.optimal_choice]:.3f}x)")
            print(f"  Worst portfolio: {optimality.worst_choice} (return: {optimality.expected_returns[optimality.worst_choice]:.3f}x)")
            
            results[mechanism] = {
                "config": config,
                "initial_state": initial_state,
                "optimality": optimality,
                "status": "ready"
            }
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[mechanism] = {"status": "error", "error": str(e)}
    
    print(f"\n✅ Enhanced mechanism comparison setup complete")
    print(f"   Ready to run simulations with {len([r for r in results.values() if r.get('status') == 'ready'])} mechanisms")
    
    return results

if __name__ == "__main__":
    # Run integration example
    run_enhanced_mechanism_comparison()