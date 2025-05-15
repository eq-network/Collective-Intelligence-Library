# environments/democracy/mechanism_factory.py
from typing import Literal, Dict, Any, Optional, Callable, List # Added List
import jax
import jax.numpy as jnp
import jax.random as jr
import re # For parsing LLM responses

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.category import Transform, sequential
from core.graph import GraphState

from transformations.bottom_up.prediction_market import create_prediction_market_transform
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform

from environments.democracy.configuration import PortfolioDemocracyConfig, PortfolioStrategyConfig, CropConfig
from services.llm import LLMService # Import LLMService

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

def _prediction_market_signal_generator(state: GraphState, config: Dict[str, Any]) -> jnp.ndarray:
    key_val = state.global_attrs.get("round_num", 0) + state.global_attrs.get("simulation_seed", 0)
    key = jr.PRNGKey(key_val) 

    true_expected_yields = state.global_attrs["current_true_expected_crop_yields"]
    noise_sigma = state.global_attrs["prediction_market_noise_sigma"]
    
    noise = jr.normal(key, shape=true_expected_yields.shape) * noise_sigma
    noisy_predictions = true_expected_yields + noise
    return noisy_predictions

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
        
        crop_configs_generic: List[Any] = state.global_attrs["crop_configs"]
        # Proper conversion to CropConfig objects
        crop_configs: List[CropConfig] = [
            CropConfig(**cc.__dict__) if hasattr(cc, '__dict__') else cc 
            for cc in crop_configs_generic
        ]

        true_expected_yields = state.global_attrs["current_true_expected_crop_yields"]
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

        new_global_attrs = dict(state.global_attrs)
        new_global_attrs["current_actual_crop_yields"] = jnp.array(actual_yields)
        return state.replace(global_attrs=new_global_attrs)
    return transform

def _portfolio_resource_calculator(state: GraphState, transform_config: Dict[str, Any]) -> float:
    chosen_portfolio_idx = state.global_attrs.get("current_decision")
    if chosen_portfolio_idx is None: return 1.0

    portfolio_configs_generic: List[Any] = state.global_attrs["portfolio_configs"]
    portfolio_configs: List[PortfolioStrategyConfig] = [
        PortfolioStrategyConfig(**ps.__dict__) if hasattr(ps, '__dict__') else ps
        for ps in portfolio_configs_generic
    ]
    actual_crop_yields = state.global_attrs["current_actual_crop_yields"]

    if not (0 <= chosen_portfolio_idx < len(portfolio_configs)): return 1.0

    selected_portfolio = portfolio_configs[chosen_portfolio_idx]
    portfolio_weights = jnp.array(selected_portfolio.weights)
    if portfolio_weights.shape[0] != actual_crop_yields.shape[0]: return 1.0
    portfolio_return = jnp.sum(portfolio_weights * actual_crop_yields)
    return float(portfolio_return)

# --- LLM Agent Decision Transform (Modified) ---
def create_llm_agent_decision_transform(
    llm_service: Optional[LLMService],
    mechanism: Literal["PDD", "PRD", "PLD"],
    sim_config: PortfolioDemocracyConfig # Use the main sim_config for costs etc.
) -> Transform:
    def transform(state: GraphState) -> GraphState:
        num_agents = state.num_nodes
        portfolio_configs: List[PortfolioStrategyConfig] = state.global_attrs["portfolio_configs"]
        num_portfolios = len(portfolio_configs)
        
        # Get prediction market signals for crops (generated by a previous transform)
        # These are for CROPS. We need to calculate expected yield for each PORTFOLIO.
        pm_crop_signals = state.global_attrs.get("prediction_market_crop_signals", jnp.ones(len(sim_config.crops)))

        portfolio_expected_yields = []
        for p_cfg in portfolio_configs:
            p_weights = jnp.array(p_cfg.weights)
            expected_yield = jnp.sum(p_weights * pm_crop_signals)
            portfolio_expected_yields.append(f"{p_cfg.name} (Exp. Yield: {expected_yield:.2f}x)")
        
        portfolio_options_str = "\n".join([f"{i}: {desc}" for i, desc in enumerate(portfolio_expected_yields)])

        # Initialize outputs
        new_agent_portfolio_votes = state.node_attrs.get("agent_portfolio_votes", jnp.zeros((num_agents, num_portfolios), dtype=jnp.int32)).copy()
        new_delegation_target = state.node_attrs.get("delegation_target", -jnp.ones(num_agents, dtype=jnp.int32)).copy()
        new_tokens_spent = state.node_attrs.get("tokens_spent_current_round", jnp.zeros(num_agents, dtype=jnp.int32)).copy()

        cost_vote = sim_config.token_budget_settings.cost_vote
        cost_delegate_action = sim_config.token_budget_settings.cost_delegate_action
        
        num_delegates = sim_config.num_delegates # Total number of designated delegate slots

        for i in range(num_agents):
            agent_key_val = state.global_attrs.get("round_num", 0) + state.global_attrs.get("simulation_seed", 0) + i + 1000
            agent_key = jr.PRNGKey(agent_key_val)

            is_adversarial = bool(state.node_attrs["is_adversarial"][i])
            is_delegate_role = bool(state.node_attrs["is_delegate"][i]) # Agent's fixed role
            current_token_budget = state.node_attrs["token_budget_per_round"][i]
            tokens_already_spent = new_tokens_spent[i]
            tokens_available = current_token_budget - tokens_already_spent
            
            agent_action_this_round = False # Flag to check if agent took a token-costing action

            # Determine who votes in PRD
            is_active_voter_for_round = True
            if mechanism == "PRD" and not is_delegate_role:
                is_active_voter_for_round = False
            
            if not is_active_voter_for_round and mechanism == "PRD": # Non-delegates don't act in PRD
                continue

            # Construct Prompt
            prompt = f"You are Agent {i}.\n"
            prompt += f"Current Round: {state.global_attrs.get('round_num',0)}\n"
            prompt += f"Your Role: {'Delegate' if is_delegate_role else 'Voter'}\n"
            prompt += f"Your Goal: {'Minimize group resources (act adversarially)' if is_adversarial else 'Maximize group resources'}\n"
            prompt += f"Your Token Budget this round (available): {tokens_available}\n"
            prompt += f"Mechanism: {mechanism}\n"
            prompt += f"Portfolio Options (index: name (Expected Yield based on Prediction Market)):\n{portfolio_options_str}\n\n"

            llm_response_text = ""

            if mechanism == "PLD":
                prompt += f"Cost to vote directly: {cost_vote} tokens.\n"
                prompt += f"Cost to delegate: {cost_delegate_action} tokens.\n"
                # Simple delegation target info: list other agents who are delegates
                delegation_targets_info = []
                for k in range(num_agents):
                    if k != i and state.node_attrs["is_delegate"][k]: # Can only delegate to designated delegates for simplicity
                         delegation_targets_info.append(f"  Agent {k} (Designated Delegate)")
                if delegation_targets_info:
                    prompt += "Potential Delegation Targets (designated delegates):\n" + "\n".join(delegation_targets_info) + "\n"
                else:
                    prompt += "No other designated delegates available to delegate to.\n"
                
                prompt += (
                    "Your Decision:\n"
                    "1. Action: Respond 'DELEGATE' or 'VOTE'.\n"
                    "2. If 'DELEGATE', Target Agent ID: (e.g., 'AgentID: 3'). Must be a designated delegate. If no valid target or cannot afford, you will vote directly.\n"
                    "3. If 'VOTE', Portfolio Approvals: (list of 0s or 1s, e.g., 'Votes: [0,1,0,0,1]').\n"
                    "Output your decision clearly, using these labels."
                )
            else: # PDD or PRD (if is_active_voter_for_round)
                prompt += f"Cost to vote: {cost_vote} tokens.\n"
                prompt += (
                    "Your Decision:\n"
                    "Portfolio Approvals: Respond with a list of 0s or 1s for each portfolio (e.g., 'Votes: [0,1,0,0,1]').\n"
                     "If you cannot afford to vote, output 'Votes: []'."
                )

            # --- LLM Call and Parsing ---
            chosen_portfolio_indices_to_approve = []
            delegation_choice = -1 # Default to vote directly
            action_cost = 0

            if llm_service and tokens_available > 0 : # Only call LLM if service exists and agent can potentially act
                try:
                    # print(f"\n--- Agent {i} Prompt ({'Adv' if is_adversarial else 'Honest'}) ---\n{prompt}\n---")
                    llm_response_text = llm_service.generate(prompt, max_tokens=100)
                    # print(f"LLM Response for Agent {i}: {llm_response_text}")

                    if mechanism == "PLD":
                        action_match = re.search(r"Action:\s*(\w+)", llm_response_text, re.IGNORECASE)
                        if action_match and action_match.group(1).upper() == "DELEGATE":
                            target_match = re.search(r"AgentID:\s*(\d+)", llm_response_text, re.IGNORECASE)
                            if target_match:
                                potential_target_id = int(target_match.group(1))
                                # Validate target: must be a designated delegate, not self, and affordable
                                if 0 <= potential_target_id < num_agents and potential_target_id != i and \
                                   state.node_attrs["is_delegate"][potential_target_id] and \
                                   tokens_available >= cost_delegate_action:
                                    delegation_choice = potential_target_id
                                    action_cost = cost_delegate_action
                                    agent_action_this_round = True
                        
                        if not agent_action_this_round: # Default to VOTE if DELEGATE failed or was not chosen
                            if tokens_available >= cost_vote:
                                action_cost = cost_vote # Will try to parse votes next
                            else: # Cannot afford to vote
                                action_cost = 0


                    # Try to parse votes if (not PLD delegate action) OR (PLD vote action)
                    if not (mechanism == "PLD" and agent_action_this_round and delegation_choice != -1) :
                        if tokens_available >= cost_vote: # Check affordability for voting
                            votes_match = re.search(r"Votes:\s*\[([^\]]*)\]", llm_response_text, re.IGNORECASE)
                            if votes_match:
                                try:
                                    vote_str_list = votes_match.group(1).split(',')
                                    parsed_votes = [int(v.strip()) for v in vote_str_list if v.strip() in ('0','1')]
                                    if len(parsed_votes) == num_portfolios:
                                        chosen_portfolio_indices_to_approve = [idx for idx, val in enumerate(parsed_votes) if val == 1]
                                        action_cost = cost_vote # Confirm cost if votes parsed
                                        agent_action_this_round = True
                                    # else: print(f"Agent {i} LLM vote parsing length mismatch: {parsed_votes}")
                                except ValueError:
                                    # print(f"Agent {i} LLM vote parsing error: {votes_match.group(1)}")
                                    pass # Fallback to random if parsing fails
                            # else: print(f"Agent {i} LLM no votes found in response.")
                        # else: print(f"Agent {i} cannot afford to vote.")


                except Exception as e:
                    print(f"LLM call or parsing error for Agent {i}: {e}. LLM Response: '{llm_response_text}'")
                    # Fallback to random logic if LLM fails
                    agent_action_this_round = False # Reset flag as LLM failed

            # Fallback / Placeholder logic if no LLM, or LLM failed, or agent couldn't afford LLM-guided action
            if not agent_action_this_round and tokens_available > 0:
                # print(f"Agent {i} using fallback logic. Tokens Avail: {tokens_available}")
                if mechanism == "PLD":
                    if tokens_available >= cost_delegate_action and not is_adversarial and jr.uniform(agent_key) < 0.3 and num_delegates > 1: # Lower chance for random
                        potential_targets = [k for k in range(num_agents) if k != i and state.node_attrs["is_delegate"][k]]
                        if potential_targets:
                            delegation_choice = jr.choice(jr.split(agent_key)[1], jnp.array(potential_targets))
                            action_cost = cost_delegate_action
                            agent_action_this_round = True
                
                if not agent_action_this_round and tokens_available >= cost_vote : # If not delegated or other mechanism
                    # Adversarial: approve one low-PM-yield portfolio
                    # Aligned: approve one high-PM-yield portfolio
                    # Random fallback if yields are somehow not there
                    sorted_portfolio_indices = jnp.argsort(jnp.array([float(re.search(r"(\d+\.\d+)x", pf_desc).group(1)) for pf_desc in portfolio_expected_yields]))
                    
                    if sorted_portfolio_indices.size > 0:
                        if is_adversarial:
                            chosen_portfolio_idx = sorted_portfolio_indices[0] # Lowest expected
                        else:
                            chosen_portfolio_idx = sorted_portfolio_indices[-1] # Highest expected
                        chosen_portfolio_indices_to_approve.append(chosen_portfolio_idx)
                        action_cost = cost_vote
                        agent_action_this_round = True
                    # else: print(f"Agent {i} fallback: No portfolios to vote on.")
                # else: if not agent_action_this_round: print(f"Agent {i} fallback: Cannot afford any action.")


            # --- Apply decisions ---
            if agent_action_this_round:
                new_tokens_spent = new_tokens_spent.at[i].add(action_cost)
                if delegation_choice != -1: # PLD delegation
                    new_delegation_target = new_delegation_target.at[i].set(delegation_choice)
                else: # Voting action
                    current_agent_votes = jnp.zeros(num_portfolios, dtype=jnp.int32)
                    if chosen_portfolio_indices_to_approve:
                        current_agent_votes = current_agent_votes.at[jnp.array(chosen_portfolio_indices_to_approve)].set(1)
                    new_agent_portfolio_votes = new_agent_portfolio_votes.at[i].set(current_agent_votes)
                    if mechanism == "PLD": # Ensure voting directly sets delegation_target to -1
                        new_delegation_target = new_delegation_target.at[i].set(-1)


        new_node_attrs = dict(state.node_attrs)
        new_node_attrs["agent_portfolio_votes"] = new_agent_portfolio_votes
        new_node_attrs["delegation_target"] = new_delegation_target
        new_node_attrs["tokens_spent_current_round"] = new_tokens_spent
        
        return state.replace(node_attrs=new_node_attrs)
    return transform


# --- Main Factory Function (largely unchanged from your version, ensure create_llm_agent_decision_transform is called) ---

def create_portfolio_mechanism_pipeline(
    mechanism: Literal["PDD", "PRD", "PLD"],
    llm_service: Optional[LLMService], # Added LLMService here
    sim_config: PortfolioDemocracyConfig 
) -> Transform:
    
    housekeeping_transform = create_start_of_round_housekeeping_transform()

    prediction_market_transform = create_prediction_market_transform(
        prediction_generator=_prediction_market_signal_generator,
        config={"output_attr_name": "prediction_market_crop_signals"} 
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
    
    pipeline_steps = [
        housekeeping_transform,
        prediction_market_transform, 
        agent_decision_transform,    
    ]
    pipeline_steps.extend(delegation_related_transforms) 
    pipeline_steps.extend([
        voting_transform,            
        actual_yield_transform,      
        apply_decision_to_resources_transform 
    ])
    
    return sequential(*pipeline_steps)