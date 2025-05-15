# For: environments/democracy/portfolio_mechanism_factory.py
# Based on input_file_7.py and requirements, integrating AnalysisAdapter

from typing import Literal, Dict, Any, Optional, Callable
import jax.numpy as jnp

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState
from core.category import Transform, sequential
# Assuming these are correctly importable from the project structure
from transformations.bottom_up.prediction_market import create_prediction_market_transform
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform

from environments.democracy.configuration import PortfolioDemocracyConfig # input_file_6.py
from services.adapters import AnalysisAdapter # From previous step

def create_portfolio_mechanism_pipeline(
    mechanism: Literal["PDD", "PRD", "PLD"],
    config: PortfolioDemocracyConfig,
    analysis_adapter: AnalysisAdapter # Changed from external_analyzer to adapter
) -> Transform:
    """
    Create a transformation pipeline for portfolio-based democratic mechanisms.
    
    Args:
        mechanism: Democratic mechanism type (PDD, PRD, or PLD).
        config: PortfolioDemocracyConfig object.
        analysis_adapter: Adapter for portfolio analysis (e.g., LLM-based or rule-based).
        
    Returns:
        A composed transformation pipeline for the specified mechanism.
    """

    # --- Portfolio Preference Aggregator for Voting Transform ---
    def portfolio_preference_aggregator(state: GraphState, transform_config: Dict[str, Any]) -> jnp.ndarray:
        # transform_config here is the one passed to create_voting_transform
        # Default to uniform distribution if preferences not available
        if "portfolio_preferences" not in state.node_attrs:
            num_portfolios = len(state.global_attrs.get("portfolios", {}))
            if num_portfolios == 0: return jnp.array([1.0]) # Should not happen with proper init
            return jnp.ones(num_portfolios) / num_portfolios
        
        preferences = state.node_attrs["portfolio_preferences"] # Shape: (num_agents, num_portfolios)
        mechanism_type = transform_config.get("mechanism_type", "direct") # This comes from voting_config
        
        if mechanism_type == "direct": # PDD
            # Weights all agents equally that have preferences
            active_voters_mask = jnp.sum(preferences, axis=1) > 0 # Consider only agents with non-zero preferences sum
            if jnp.sum(active_voters_mask) == 0: # No active voters
                 return jnp.ones(preferences.shape[1]) / preferences.shape[1]
            return jnp.sum(preferences * active_voters_mask[:, jnp.newaxis], axis=0) / jnp.sum(active_voters_mask)

        elif mechanism_type == "representative": # PRD
            if "is_representative" not in state.node_attrs: # Fallback to direct if not defined
                print("Warning: 'is_representative' not in node_attrs for PRD. Falling back to direct aggregation.")
                return jnp.mean(preferences, axis=0) # Or use the PDD logic above
            
            is_rep = state.node_attrs["is_representative"].astype(jnp.float32) # Ensure float for multiplication
            num_reps = jnp.sum(is_rep)
            if num_reps == 0: # No representatives
                print("Warning: No representatives found for PRD. Returning uniform distribution.")
                return jnp.ones(preferences.shape[1]) / preferences.shape[1]
            
            # Mask non-representatives' preferences
            rep_preferences = preferences * is_rep[:, jnp.newaxis]
            return jnp.sum(rep_preferences, axis=0) / num_reps
        
        elif mechanism_type in ["liquid", "weighted"]: # PLD
            if "voting_power" not in state.node_attrs: # Fallback if no voting power
                print("Warning: 'voting_power' not in node_attrs for PLD. Falling back to direct aggregation.")
                return jnp.mean(preferences, axis=0) # Or use PDD logic

            voting_power = state.node_attrs["voting_power"] # Shape: (num_agents,)
            weighted_preferences = preferences * voting_power[:, jnp.newaxis] # Element-wise multiplication broadcasts voting_power
            
            portfolio_scores = jnp.sum(weighted_preferences, axis=0)
            total_score_sum = jnp.sum(portfolio_scores)
            
            if total_score_sum <= 1e-9: # Avoid division by zero / very small numbers
                return jnp.ones_like(portfolio_scores) / portfolio_scores.shape[0]
            return portfolio_scores / total_score_sum
            
        # Fallback for unknown mechanism_type (should not be reached if config is correct)
        print(f"Warning: Unknown mechanism_type '{mechanism_type}' in aggregator. Using direct mean.")
        return jnp.mean(preferences, axis=0)

    # --- Portfolio Resource Calculator for Resource Transform ---
    def portfolio_resource_calculator(state: GraphState, transform_config: Dict[str, Any]) -> float:
        # transform_config here is the one passed to create_resource_transform
        if "current_decision" not in state.global_attrs or "portfolios" not in state.global_attrs:
            # print("Warning: current_decision or portfolios not in global_attrs. No resource change.")
            return 1.0 # Multiplier: No change

        decision_idx = state.global_attrs["current_decision"]
        portfolios_map = state.global_attrs["portfolios"] # Dict: name -> portfolio_data
        portfolio_names = list(portfolios_map.keys()) # Maintain order for indexing
        
        if not (0 <= decision_idx < len(portfolio_names)):
            # print(f"Warning: Invalid decision_idx {decision_idx}. No resource change.")
            return 1.0

        selected_portfolio_name = portfolio_names[decision_idx]
        selected_portfolio_data = portfolios_map[selected_portfolio_name]
        
        portfolio_weights = jnp.array(selected_portfolio_data.get("weights", []))
        
        # Use true returns for asset performance evaluation
        # These should be updated each round in a more dynamic simulation
        true_asset_returns = state.global_attrs.get("true_returns") 
        if true_asset_returns is None or len(portfolio_weights) != len(true_asset_returns):
            # print("Warning: true_returns not available or mismatched. Using prediction_market or default.")
            # Fallback to prediction market if true returns are missing, or default to no change
            true_asset_returns = state.global_attrs.get("prediction_market", jnp.ones_like(portfolio_weights))
            if len(portfolio_weights) != len(true_asset_returns): # Final fallback
                 return 1.0


        # Calculate the actual return multiplier for the collective's resources
        actual_return_multiplier = jnp.sum(portfolio_weights * true_asset_returns)
        return float(actual_return_multiplier)

    # --- Token Budget Refresh Calculator for Resource Transform ---
    def token_budget_refresh_calculator(state: GraphState, transform_config: Dict[str, Any]) -> GraphState:
        # This calculator is different: it modifies state directly and returns the new state.
        # It's used with create_resource_transform for its structure, but behavior is state update.
        if not config.token_system.enabled or \
           "token_budget" not in state.node_attrs or \
           "tokens_spent" not in state.node_attrs:
            return state # No change if token system is off or attrs missing

        current_round = state.global_attrs.get("round", 0)
        refresh_period = config.token_system.refresh_period # from PortfolioDemocracyConfig
        
        # Refresh at the START of rounds that are multiples of refresh_period (but not round 0)
        if current_round > 0 and current_round % refresh_period == 0:
            new_node_attrs = dict(state.node_attrs)
            new_node_attrs["tokens_spent"] = jnp.zeros_like(state.node_attrs["tokens_spent"])
            # Also update current_period in global_attrs if tracking periods
            new_global_attrs = dict(state.global_attrs)
            new_global_attrs["current_period"] = state.global_attrs.get("current_period",0) + 1
            return state.replace(node_attrs=new_node_attrs, global_attrs=new_global_attrs)
        
        return state # No change to tokens_spent

    # This structure is for a transform that returns a float (multiplier).
    # For state update, we need a direct transform or adapt create_resource_transform.
    # Let's make a direct token_refresh_transform.
    def token_refresh_transform_fn(state: GraphState) -> GraphState:
        return token_budget_refresh_calculator(state, {})


    # --- Portfolio Analysis Transform (incorporates the adapter) ---
    def analyze_portfolios_transform(state: GraphState) -> GraphState:
        if not config.token_system.enabled and "token_budget" not in state.node_attrs : # Allow analysis if no token system
             pass # proceed to analysis even if no token system
        elif "portfolios" not in state.global_attrs:
            # print("Warning: Portfolios not defined in global_attrs. Skipping analysis.")
            return state
        
        portfolios_map = state.global_attrs["portfolios"]
        num_agents = state.num_nodes
        
        # Initialize preferences (e.g., with zeros or current values if iterative)
        # For now, assume each analysis overwrites previous preferences for simplicity
        new_portfolio_preferences = jnp.zeros((num_agents, len(portfolios_map)))
        
        # Handle token spending updates
        # Need to copy tokens_spent to avoid modifying the input `state`'s array directly
        # if it's used elsewhere before `state.replace`. JAX arrays are immutable,
        # but dicts holding them are not.
        current_tokens_spent = state.node_attrs.get("tokens_spent", jnp.zeros(num_agents))
        updated_tokens_spent_list = list(current_tokens_spent) # Modifiable list

        for i in range(num_agents):
            agent_node_attrs = {attr: val[i] for attr, val in state.node_attrs.items()}
            # Pass a copy of global_attrs to avoid unintended modifications by adapter if any
            # Though adapters should also follow immutability if they create new states.
            agent_global_info = dict(state.global_attrs) 

            # Call the injected analysis adapter's method
            # analysis_adapter comes from the outer scope of create_portfolio_mechanism_pipeline
            analysis_result = analysis_adapter.analyze(agent_node_attrs, agent_global_info)
            
            agent_prefs = analysis_result.get("preferences", {})
            tokens_used = analysis_result.get("tokens_spent", 0.0)

            for j, p_name in enumerate(portfolios_map.keys()):
                if p_name in agent_prefs:
                    new_portfolio_preferences = new_portfolio_preferences.at[i, j].set(agent_prefs[p_name])
            
            updated_tokens_spent_list[i] = updated_tokens_spent_list[i] + tokens_used
            
        final_node_attrs = dict(state.node_attrs)
        final_node_attrs["portfolio_preferences"] = new_portfolio_preferences
        if "tokens_spent" in final_node_attrs or config.token_system.enabled: # Only update if relevant
            final_node_attrs["tokens_spent"] = jnp.array(updated_tokens_spent_list)
            
        return state.replace(node_attrs=final_node_attrs)

    # --- Constructing the Pipeline ---
    # Common initial transform: Prediction Market (updates asset predictions)
    # The prediction market in this context might be for the underlying assets,
    # not the portfolios themselves, which agents then use for portfolio analysis.
    asset_prediction_market_transform = create_prediction_market_transform(
        # prediction_generator could be configured to model asset price movements
        # For now, assume it uses config.market.accuracy and initial_predictions to generate noisy signals
        config={
            "accuracy": config.market.accuracy,
            "noise_level": config.market.noise_level, # Add this if create_prediction_market_transform supports it
            "num_options": len(config.resources.asset_names) 
        }
    )
    
    # Token refresh transform (direct function)
    token_refresh_transform = token_refresh_transform_fn
    
    # Resource application transform
    apply_resources_transform = create_resource_transform(
        resource_calculator=portfolio_resource_calculator,
        config={"track_history": config.track_metrics.get("resource_history",True)} # Pass from main config
    )

    # Voting transform
    # The mechanism_type for aggregator is specific to the pipeline (PDD, PRD, PLD)
    # It will be set when creating the voting_transform for each pipeline.

    # Mechanism-specific pipelines
    if mechanism == "PDD":
        pdd_voting_transform = create_voting_transform(
            vote_aggregator=portfolio_preference_aggregator,
            config={"mechanism_type": "direct"} # PDD uses direct aggregation
        )
        return sequential(
            asset_prediction_market_transform,
            token_refresh_transform,
            analyze_portfolios_transform,
            pdd_voting_transform,
            apply_resources_transform
        )
    
    elif mechanism == "PRD":
        prd_voting_transform = create_voting_transform(
            vote_aggregator=portfolio_preference_aggregator,
            config={"mechanism_type": "representative"} # PRD uses representative aggregation
        )
        return sequential(
            asset_prediction_market_transform,
            token_refresh_transform,
            analyze_portfolios_transform,
            prd_voting_transform, # Representatives vote
            apply_resources_transform
        )
    
    elif mechanism == "PLD":
        pld_delegation_transform = create_delegation_transform() # Assumes 'delegation_choices' in node_attrs
        pld_power_flow_transform = create_power_flow_transform()
        pld_voting_transform = create_voting_transform(
            vote_aggregator=portfolio_preference_aggregator,
            config={"mechanism_type": "liquid"} # PLD uses weighted/liquid aggregation
        )
        return sequential(
            asset_prediction_market_transform,
            token_refresh_transform,
            analyze_portfolios_transform, # Agents form preferences
            pld_delegation_transform,     # Agents update delegations based on new info (needs agent logic for choices)
            pld_power_flow_transform,     # Calculate voting power
            pld_voting_transform,         # Vote with updated power
            apply_resources_transform
        )
    
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

if __name__ == '__main__':
    from environments.democracy.portfolio_initialization import initialize_portfolio_state
    from services.adapters import RuleBasedAnalysisAdapter, LLMAnalysisAdapter
    from services.llm import MockLLMService # Assuming MockLLMService is defined in llm.py for testing

    # Create a sample config
    test_config = PortfolioDemocracyConfig(
        num_agents=5, 
        num_rounds=2,
        mechanism="PLD",
        resources = PortfolioDemocracyConfig().resources.replace(asset_names=["AssetX", "AssetY"]),
        market = PortfolioDemocracyConfig().market.replace(initial_predictions=[1.1, 0.9], true_returns=[1.2, 0.8]),
        token_system = PortfolioDemocracyConfig().token_system.replace(refresh_period=1) # Refresh every round
    )
    
    prng_key = jax.random.PRNGKey(test_config.seed)
    initial_gs = initialize_portfolio_state(test_config, prng_key)
    
    print(f"Initial state created for {test_config.mechanism} with {test_config.num_agents} agents.")
    print(f"Initial Tokens Spent: {initial_gs.node_attrs['tokens_spent']}")
    print(f"Initial Total Resources: {initial_gs.global_attrs['total_resources']}")

    # Test with RuleBasedAdapter
    rule_adapter = RuleBasedAnalysisAdapter()
    pld_pipeline_rule = create_portfolio_mechanism_pipeline("PLD", test_config, rule_adapter)
    
    print("\nRunning PLD pipeline with RuleBasedAdapter...")
    current_state_rule = initial_gs
    for r in range(1, test_config.num_rounds + 1):
        current_state_rule = current_state_rule.replace(global_attrs={**current_state_rule.global_attrs, "round":r})
        current_state_rule = pld_pipeline_rule(current_state_rule) # Apply one step of the pipeline
        print(f"Round {r} (Rule): Tokens Spent: {current_state_rule.node_attrs['tokens_spent']}, Resources: {current_state_rule.global_attrs['total_resources']:.2f}, Decision: {current_state_rule.global_attrs.get('current_decision')}")

    # Test with LLMAdapter (using Mock)
    mock_llm_service = MockLLMService() # Defined in adapters.py test section
    llm_adapter = LLMAnalysisAdapter(llm_service=mock_llm_service)
    pld_pipeline_llm = create_portfolio_mechanism_pipeline("PLD", test_config, llm_adapter)

    print("\nRunning PLD pipeline with LLMAdapter (Mock)...")
    current_state_llm = initial_gs
    for r in range(1, test_config.num_rounds + 1):
        current_state_llm = current_state_llm.replace(global_attrs={**current_state_llm.global_attrs, "round":r})
        current_state_llm = pld_pipeline_llm(current_state_llm)
        print(f"Round {r} (LLM): Tokens Spent: {current_state_llm.node_attrs['tokens_spent']}, Resources: {current_state_llm.global_attrs['total_resources']:.2f}, Decision: {current_state_llm.global_attrs.get('current_decision')}")

    # Test PDD
    pdd_pipeline_rule = create_portfolio_mechanism_pipeline("PDD", test_config, rule_adapter)
    print("\nRunning PDD pipeline with RuleBasedAdapter...")
    current_state_pdd = initial_gs
    for r in range(1, test_config.num_rounds + 1):
        current_state_pdd = current_state_pdd.replace(global_attrs={**current_state_pdd.global_attrs, "round":r})
        current_state_pdd = pdd_pipeline_rule(current_state_pdd)
        print(f"Round {r} (PDD): Tokens Spent: {current_state_pdd.node_attrs['tokens_spent']}, Resources: {current_state_pdd.global_attrs['total_resources']:.2f}, Decision: {current_state_pdd.global_attrs.get('current_decision')}")
    
    # Test PRD (Note: is_representative needs to be properly set in initialization for PRD to work as intended)
    # For this test, it will likely default due to no reps or fallback in aggregator
    test_config_prd = test_config.replace(mechanism="PRD")
    initial_gs_prd = initialize_portfolio_state(test_config_prd, prng_key) # Re-init with PRD mechanism for correct is_representative
    
    prd_pipeline_rule = create_portfolio_mechanism_pipeline("PRD", test_config_prd, rule_adapter)
    print("\nRunning PRD pipeline with RuleBasedAdapter...")
    current_state_prd = initial_gs_prd
    for r in range(1, test_config_prd.num_rounds + 1):
        current_state_prd = current_state_prd.replace(global_attrs={**current_state_prd.global_attrs, "round":r})
        current_state_prd = prd_pipeline_rule(current_state_prd)
        print(f"Round {r} (PRD): Tokens Spent: {current_state_prd.node_attrs['tokens_spent']}, Resources: {current_state_prd.global_attrs['total_resources']:.2f}, Decision: {current_state_prd.global_attrs.get('current_decision')}")
        if r==1: print(f"  (PRD Test) Representatives: {current_state_prd.node_attrs['is_representative']}")

print("environments/democracy/portfolio_mechanism_factory.py content generated.")