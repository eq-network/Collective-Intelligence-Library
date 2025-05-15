# environments/democracy/mechanism_factory.py
from typing import Literal, Dict, Any, Optional, Callable
import jax.numpy as jnp

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
from environments.democracy.configuration import PortfolioDemocracyConfig

def create_portfolio_mechanism_pipeline(
    mechanism: Literal["PDD", "PRD", "PLD"],
    config: PortfolioDemocracyConfig,
    external_analyzer: Optional[Callable] = None
) -> Transform:
    """
    Create a transformation pipeline for portfolio-based democratic mechanisms.
    
    Args:
        mechanism: Democratic mechanism type (PDD, PRD, or PLD)
        config: Portfolio democracy configuration
        external_analyzer: Optional function for advanced portfolio analysis
        
    Returns:
        A composed transformation pipeline for the specified mechanism
    """
    # Define portfolio analyzer transform
    def create_portfolio_analyzer() -> Transform:
        """
        Creates a transform that analyzes portfolios and updates agent preferences.
        
        This implements the core decision-making logic for agents based on their
        goals (aligned or adversarial) and available information.
        """
        def analyze_portfolios(state: GraphState) -> GraphState:
            # Skip if portfolios not defined
            if "portfolios" not in state.global_attrs:
                return state
            
            # Extract portfolio information
            portfolios = state.global_attrs["portfolios"]
            num_agents = state.num_nodes
            portfolio_preferences = jnp.zeros((num_agents, len(portfolios)))
            
            # Setup token tracking if applicable
            new_tokens_spent = None
            if "token_budget" in state.node_attrs and "tokens_spent" in state.node_attrs:
                new_tokens_spent = state.node_attrs["tokens_spent"].copy()
            
            # Process each agent
            for i in range(num_agents):
                # Extract agent attributes
                is_adversarial = bool(state.node_attrs.get("is_adversarial", jnp.zeros(num_agents))[i])
                
                # Calculate available tokens
                tokens_available = float('inf')
                if new_tokens_spent is not None:
                    token_budget = state.node_attrs["token_budget"][i]
                    tokens_spent = state.node_attrs["tokens_spent"][i]
                    tokens_available = token_budget - tokens_spent
                
                # Use external analyzer if available and affordable
                preferences = {}
                tokens_used = 10  # Base cost for voting as specified in thesis
                
                if external_analyzer and tokens_available >= tokens_used:
                    try:
                        # Prepare contexts for analysis
                        agent_context = {
                            "agent_id": i,
                            "is_adversarial": is_adversarial,
                            "tokens_available": tokens_available
                        }
                        
                        env_context = {
                            "portfolios": portfolios,
                            "prediction_market": state.global_attrs.get("prediction_market"),
                            "round": state.global_attrs.get("round", 0)
                        }
                        
                        # Call external analyzer
                        result = external_analyzer(agent_context, env_context)
                        if isinstance(result, dict) and "preferences" in result:
                            preferences = result["preferences"]
                            tokens_used = min(tokens_available, result.get("tokens_used", tokens_used))
                    except Exception:
                        # Fallback on error
                        preferences = {}
                
                # Use internal analysis if needed
                if not preferences:
                    for j, (name, portfolio) in enumerate(portfolios.items()):
                        # Extract expected return from prediction market
                        prediction_market = state.global_attrs.get("prediction_market", [1.0, 1.0, 1.0])
                        weights = portfolio.get("weights", [0.33, 0.33, 0.34])
                        
                        # Calculate expected portfolio return
                        expected_return = sum(w*p for w, p in zip(weights, prediction_market))
                        
                        # Binary approval voting as specified in thesis
                        if is_adversarial:
                            # Adversarial agents vote to minimize resources
                            # They approve portfolios with lower expected returns
                            portfolio_preferences = portfolio_preferences.at[i, j].set(
                                1.0 if expected_return < 1.0 else 0.0
                            )
                        else:
                            # Aligned agents vote to maximize resources
                            # They approve portfolios with higher expected returns
                            portfolio_preferences = portfolio_preferences.at[i, j].set(
                                1.0 if expected_return > 1.0 else 0.0
                            )
                
                # Update token usage
                if new_tokens_spent is not None:
                    new_tokens_spent = new_tokens_spent.at[i].add(tokens_used)
            
            # Create updated state
            new_node_attrs = dict(state.node_attrs)
            new_node_attrs["portfolio_preferences"] = portfolio_preferences
            
            if new_tokens_spent is not None:
                new_node_attrs["tokens_spent"] = new_tokens_spent
            
            return state.replace(node_attrs=new_node_attrs)
        
        return analyze_portfolios

    # Define the portfolio preference aggregator function
    def portfolio_preference_aggregator(state: GraphState, transform_config: Dict[str, Any]) -> jnp.ndarray:
        """
        Aggregate portfolio preferences based on mechanism type.
        
        This function is passed to the voting transform and handles different
        preference aggregation schemes based on the democratic mechanism.
        """
        # Default to uniform distribution if preferences not available
        if "portfolio_preferences" not in state.node_attrs:
            num_portfolios = len(state.global_attrs.get("portfolios", {}))
            if num_portfolios == 0:
                return jnp.array([1.0])
            return jnp.ones(num_portfolios) / num_portfolios
        
        # Extract preferences and determine aggregation strategy
        preferences = state.node_attrs["portfolio_preferences"]
        mechanism_type = transform_config.get("mechanism_type", "direct")
        
        # Direct democracy: equal weighting of all preferences
        if mechanism_type == "direct":
            # Sum binary approvals for each portfolio as specified in thesis
            return jnp.sum(preferences, axis=0)
        
        # Representative democracy: only count designated representatives
        elif mechanism_type == "representative":
            if "is_representative" not in state.node_attrs:
                return jnp.sum(preferences, axis=0)
            
            # Filter to representatives' preferences
            is_rep = state.node_attrs["is_representative"]
            rep_preferences = preferences * is_rep[:, jnp.newaxis]  # Mask non-representatives
            
            # Return sum of binary approvals from representatives
            return jnp.sum(rep_preferences, axis=0)
        
        # Liquid democracy: weight by delegated voting power
        elif mechanism_type in ["liquid", "weighted"]:
            if "voting_power" not in state.node_attrs:
                return jnp.sum(preferences, axis=0)
            
            # Apply voting power weights
            voting_power = state.node_attrs["voting_power"]
            weighted_preferences = preferences * voting_power[:, jnp.newaxis]
            
            # Sum weighted binary approvals
            return jnp.sum(weighted_preferences, axis=0)
            
        # Default fallback
        return jnp.sum(preferences, axis=0)
    
    # Define the portfolio resource calculator function
    def portfolio_resource_calculator(state: GraphState, transform_config: Dict[str, Any]) -> float:
        """
        Calculate resource changes based on the selected portfolio performance.
        
        This function computes the resource multiplier based on portfolio weights
        and actual asset returns.
        """
        # Handle missing state components
        if "current_decision" not in state.global_attrs or "portfolios" not in state.global_attrs:
            return 1.0  # No change if insufficient information
        
        # Extract decision and portfolio data
        decision_idx = state.global_attrs["current_decision"]
        portfolios = state.global_attrs["portfolios"]
        portfolio_names = list(portfolios.keys())
        
        # Validate decision index
        if decision_idx < 0 or decision_idx >= len(portfolio_names):
            return 1.0  # No change for invalid decision
        
        # Extract selected portfolio
        portfolio_name = portfolio_names[decision_idx]
        portfolio = portfolios[portfolio_name]
        
        # Get portfolio weights
        weights = jnp.array(portfolio.get("weights", [0.33, 0.33, 0.34]))
        
        # Use true returns if available, otherwise prediction market
        true_returns = state.global_attrs.get("true_returns")
        if true_returns is None:
            true_returns = state.global_attrs.get("prediction_market", jnp.ones_like(weights))
        
        # Calculate weighted return as the resource multiplier
        return float(jnp.sum(weights * true_returns))
    
    # Correct token budget refresh function
    def token_budget_refresh(state: GraphState, transform_config: Dict[str, Any]) -> GraphState:
        """
        Reset token spending counters at the start of each budget period.
        """
        # Skip if token system not active
        if "token_budget" not in state.node_attrs or "tokens_spent" not in state.node_attrs:
            return state
        
        # Check if we need to refresh tokens
        current_round = state.global_attrs.get("round", 0)
        refresh_period = transform_config.get("refresh_period", 5)
        
        if current_round % refresh_period == 0 and current_round > 0:
            # Reset token spending
            new_node_attrs = dict(state.node_attrs)
            new_node_attrs["tokens_spent"] = jnp.zeros_like(state.node_attrs["tokens_spent"])
            return state.replace(node_attrs=new_node_attrs)
        
        return state
    
    # Create a standard version of the token refresh transform that returns a float
    def token_refresh_calculator(state: GraphState, transform_config: Dict[str, Any]) -> float:
        """
        Token refresh calculator that returns a float multiplier (always 1.0).
        
        This is used to create a compatible transform for the resource calculator,
        which expects to return a float multiplier.
        """
        # Apply the actual token refresh logic
        token_budget_refresh(state, transform_config)
        
        # Return a neutral multiplier since this transform doesn't modify resources
        return 1.0
    
    # Create prediction market transform
    prediction_market = create_prediction_market_transform(
        config={
            "accuracy": getattr(config.market, "accuracy", 0.7),
            "num_options": len(getattr(config.market, "initial_predictions", [1.0, 1.0, 1.0]))
        }
    )
    
    # Create token refresh transform
    token_refresh = create_resource_transform(
        resource_calculator=token_refresh_calculator,
        config={"refresh_period": getattr(config.token_system, "refresh_period", 5)}
    )
    
    # Create portfolio analyzer transform
    portfolio_analyzer = create_portfolio_analyzer()
    
    # Create resource transform
    resource_transform = create_resource_transform(
        resource_calculator=portfolio_resource_calculator,
        config={"track_history": True}
    )
    
    # Construct mechanism-specific pipeline
    if mechanism == "PDD":
        # Predictive Direct Democracy
        voting = create_voting_transform(
            vote_aggregator=portfolio_preference_aggregator,
            config={"mechanism_type": "direct"}
        )
        
        return sequential(
            prediction_market,  # Update prediction market
            token_refresh,      # Refresh token budgets
            portfolio_analyzer, # Analyze portfolios and update preferences
            voting,             # Aggregate votes
            resource_transform  # Update resources based on decision
        )
    
    elif mechanism == "PRD":
        # Predictive Representative Democracy
        voting = create_voting_transform(
            vote_aggregator=portfolio_preference_aggregator,
            config={"mechanism_type": "representative"}
        )
        
        return sequential(
            prediction_market,  # Update prediction market
            token_refresh,      # Refresh token budgets
            portfolio_analyzer, # Analyze portfolios and update preferences
            voting,             # Aggregate votes (representative only)
            resource_transform  # Update resources based on decision
        )
    
    elif mechanism == "PLD":
        # Predictive Liquid Democracy
        delegation = create_delegation_transform()
        power_flow = create_power_flow_transform()
        
        voting = create_voting_transform(
            vote_aggregator=portfolio_preference_aggregator,
            config={"mechanism_type": "liquid"}
        )
        
        return sequential(
            prediction_market,  # Update prediction market
            token_refresh,      # Refresh token budgets
            delegation,         # Update delegation graph
            power_flow,         # Calculate voting power from delegations
            portfolio_analyzer, # Analyze portfolios and update preferences
            voting,             # Aggregate votes (weighted by delegation)
            resource_transform  # Update resources based on decision
        )
    
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")