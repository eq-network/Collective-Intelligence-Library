from typing import Literal, Dict, Any, Optional, Callable, List
import jax
import jax.numpy as jnp
import jax.random as jr
import re

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.category import Transform, sequential
from core.graph import GraphState

# Enhanced imports
from environments.democracy.configuration import (
    PortfolioDemocracyConfig, 
    PromptConfig,
    PortfolioStrategyConfig,
    CropConfig,
    create_thesis_baseline_config
)
from environments.democracy.optimality_analysis import (
    OptimalityCalculator,
    PerformanceAnalyzer,
    calculate_optimality_for_state,
    generate_optimality_prompt_text
)
from services.llm import LLMService

# Import existing transforms (unchanged)
from transformations.bottom_up.prediction_market import create_prediction_market_transform
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform
from transformations.top_down.democratic_transforms.election import create_election_transform

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
    
    # DEBUG: Add this logging
    print(f"PREDICTION_MARKET_DEBUG: Round {state.global_attrs.get('round_num', 0)}")
    print(f"  True expected yields: {true_expected_yields}")
    print(f"  Noise sigma: {noise_sigma}")
    print(f"  Generated noise: {noise}")
    print(f"  Final prediction signals: {noisy_predictions}")
    
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
        
        crop_configs_generic = state.global_attrs["crop_configs"]
        # Simple conversion to CropConfig objects
        crop_configs = [
            CropConfig(**cc.__dict__) if hasattr(cc, '__dict__') else cc 
            for cc in crop_configs_generic
        ]

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
        uniform_signals = state.global_attrs.get("prediction_market_crop_signals", jnp.ones(len(sim_config.crops)))
        
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

        # Per-agent loop
        for i in range(num_agents):
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

            # Use agent-specific prediction signals if available
            if i in agent_specific_signals:
                agent_pm_signals = agent_specific_signals[i]
            else:
                agent_pm_signals = uniform_signals

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
            prompt_result = sim_config.prompt_settings.generate_prompt(
                agent_id=i,
                round_num=state.global_attrs.get('round_num', 0),
                is_delegate=is_delegate_role,
                is_adversarial=is_adversarial,
                cognitive_resources=cognitive_resources,
                mechanism=mechanism,
                portfolio_options_str=portfolio_options_str,
                delegate_targets_str=delegate_targets_info
            )
            
            prompt = prompt_result["prompt"]
            max_tokens = prompt_result["max_tokens"]
            
            # LLM decision-making logic with ENHANCED debugging
            llm_response_text = ""
            chosen_portfolio_indices_to_approve = []
            delegation_choice = -1
            action_cost = 0
            agent_action_this_round = False

            if llm_service:
                try:
                    llm_response_text = llm_service.generate(prompt, max_tokens=max_tokens)

                    # ENHANCED DEBUG: Show meaningful LLM response snippets
                    response_snippet = llm_response_text[:100] + "..." if len(llm_response_text) > 100 else llm_response_text
                    agent_role = "Delegate" if is_delegate_role else "Voter"
                    agent_type = "Adversarial" if is_adversarial else "Aligned"
                    
                    if mechanism == "PLD":
                        action_match = re.search(r"Action:\s*(\w+)", llm_response_text, re.IGNORECASE)
                        if action_match and action_match.group(1).upper() == "DELEGATE":
                            target_match = re.search(r"AgentID:\s*(\d+)", llm_response_text, re.IGNORECASE)
                            if target_match:
                                potential_target_id = int(target_match.group(1))
                                if (0 <= potential_target_id < num_agents and 
                                   potential_target_id != i and 
                                   state.node_attrs["is_delegate"][potential_target_id]):
                                    delegation_choice = potential_target_id
                                    agent_action_this_round = True
                                    decision_summary.append(f"A{i}({agent_type[0]}{agent_role[0]}) → DELEGATE to A{potential_target_id}")

                    if not (mechanism == "PLD" and agent_action_this_round):
                        votes_match = re.search(r"Votes:\s*\[([^\]]*)\]", llm_response_text, re.IGNORECASE)
                        if votes_match:
                            try:
                                vote_str_list = votes_match.group(1).split(',')
                                parsed_votes = [int(v.strip()) for v in vote_str_list if v.strip().isdigit()]
                                if len(parsed_votes) == num_portfolios:
                                    chosen_portfolio_indices_to_approve = [idx for idx, val in enumerate(parsed_votes) if val == 1]
                                    if chosen_portfolio_indices_to_approve:
                                        portfolio_names = [portfolio_configs[idx].name for idx in chosen_portfolio_indices_to_approve]
                                        decision_summary.append(f"A{i}({agent_type[0]}{agent_role[0]}) → VOTE {portfolio_names}")
                            except ValueError:
                                pass
                        agent_action_this_round = True

                except Exception as e:
                    print(f"[R{round_num:02d}] LLM error for Agent {i}: {e}")
                    agent_action_this_round = False
            
            # Fallback logic
            if not llm_service and not agent_action_this_round:
                agent_action_this_round = True
                if mechanism == "PLD":
                    delegation_choice = -1

            # Apply decisions
            if agent_action_this_round:
                new_tokens_spent = new_tokens_spent.at[i].add(action_cost)

                if delegation_choice != -1 and mechanism == "PLD":
                    new_delegation_target = new_delegation_target.at[i].set(delegation_choice)
                    new_agent_portfolio_votes = new_agent_portfolio_votes.at[i].set(jnp.zeros(num_portfolios, dtype=jnp.int32))
                else:
                    current_agent_votes = jnp.zeros(num_portfolios, dtype=jnp.int32)
                    if chosen_portfolio_indices_to_approve:
                        current_agent_votes = current_agent_votes.at[jnp.array(chosen_portfolio_indices_to_approve)].set(1)
                    new_agent_portfolio_votes = new_agent_portfolio_votes.at[i].set(current_agent_votes)
                    if mechanism == "PLD":
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


# --- Main Factory Function (largely unchanged from your version, ensure create_llm_agent_decision_transform is called) ---

def create_portfolio_mechanism_pipeline(
    mechanism: Literal["PDD", "PRD", "PLD"],
    llm_service: Optional[LLMService], # Added LLMService here
    sim_config: PortfolioDemocracyConfig 
) -> Transform:
    
    housekeeping_transform = create_start_of_round_housekeeping_transform()

    # Initialize election_transform_prd to None so it's always bound
    election_transform_prd = None 

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
        actual_yield_transform,      
        apply_decision_to_resources_transform 
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
        config = create_enhanced_thesis_baseline_config(
            mechanism=mechanism,
            adversarial_proportion_total=0.3,
            seed=12345,
            use_redteam_prompts=True,
            include_optimality_analysis=True
        )
        
        print(f"  Configuration: {config.num_agents} agents, {config.num_delegates} delegates")
        print(f"  Red team prompts: {config.use_redteam_prompts}")
        print(f"  Optimality analysis: {config.include_optimality_analysis}")
        
        # Initialize simulation state
        key = jr.PRNGKey(config.seed)
        from environments.democracy.initialization import initialize_portfolio_democracy_graph_state
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