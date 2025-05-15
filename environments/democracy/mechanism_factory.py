# environments/democracy/portfolio_mechanism_factory.py
from typing import Literal, Dict, Any, Optional, Callable
import jax.numpy as jnp

from core.category import Transform, sequential
from core.graph import GraphState
from transformations.bottom_up.prediction_market import create_prediction_market_transform
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform
from environments.democracy.configuration import PortfolioConfig

def create_portfolio_mechanism_pipeline(
    mechanism: Literal["PDD", "PRD", "PLD"],
    config: PortfolioConfig,
    external_analyzer: Optional[Callable] = None
) -> Transform:
    """
    Create a transformation pipeline for portfolio-based democratic mechanisms.
    
    Args:
        mechanism: Democratic mechanism type (PDD, PRD, or PLD)
        config: Portfolio configuration
        external_analyzer: Optional function for advanced portfolio analysis
        
    Returns:
        A composed transformation pipeline for the specified mechanism
    """
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
            return jnp.mean(preferences, axis=0)
        
        # Representative democracy: only count designated representatives
        elif mechanism_type == "representative":
            if "is_representative" not in state.node_attrs:
                return jnp.mean(preferences, axis=0)
            
            # Filter to representatives' preferences
            is_rep = state.node_attrs["is_representative"]
            rep_preferences = preferences * is_rep[:, jnp.newaxis]  # Mask non-representatives
            rep_count = jnp.sum(is_rep)
            
            # Avoid division by zero
            return jnp.sum(rep_preferences, axis=0) / jnp.maximum(rep_count, 1.0)
        
        # Liquid democracy: weight by delegated voting power
        elif mechanism_type in ["liquid", "weighted"]:
            if "voting_power" not in state.node_attrs:
                return jnp.mean(preferences, axis=0)
            
            # Apply voting power weights
            voting_power = state.node_attrs["voting_power"]
            weighted_preferences = preferences * voting_power[:, jnp.newaxis]
            
            # Calculate weighted distribution
            weighted_sum = jnp.sum(weighted_preferences, axis=0)
            total = jnp.sum(weighted_sum)
            
            # Ensure valid probability distribution
            return weighted_sum / jnp.maximum(total, 1e-10)
            
        # Default fallback
        return jnp.mean(preferences, axis=0)
    
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
        weights = jnp.array(portfolio["weights"])
        
        # Use true returns if available, otherwise prediction market
        true_returns = state.global_attrs.get("true_returns")
        if true_returns is None:
            true_returns = state.global_attrs.get("prediction_market", jnp.ones_like(weights))
        
        # Calculate weighted return as the resource multiplier
        return float(jnp.sum(weights * true_returns))
    
    # Define token budget refresh function
    def token_budget_calculator(state: GraphState, transform_config: Dict[str, Any]) -> GraphState:
        """
        Calculate token budget refreshes at specified intervals.
        
        This function resets token spending counters at the start of each budget period.
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
    
    # Define portfolio analyzer function
    def analyze_portfolios(state: GraphState) -> GraphState:
        """
        Analyze portfolios and update agent preferences.
        
        This transformation generates preference scores for each portfolio
        based on agent characteristics and available information.
        """
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
            tokens_used = 10  # Base cost
            
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
                for name, portfolio in portfolios.items():
                    expected_return = portfolio.get("expected_return", 0.0)
                    
                    # Calculate preference score
                    if is_adversarial:
                        # Adversarial agents prefer lower expected returns
                        score = 10.0 - min(10.0, max(0.0, expected_return * 5.0))
                    else:
                        # Regular agents prefer higher expected returns
                        score = min(10.0, max(0.0, expected_return * 5.0))
                    
                    preferences[name] = score
            
            # Update preference matrix
            for j, name in enumerate(portfolios.keys()):
                if name in preferences:
                    portfolio_preferences = portfolio_preferences.at[i, j].set(preferences[name])
            
            # Update token usage
            if new_tokens_spent is not None:
                new_tokens_spent = new_tokens_spent.at[i].add(tokens_used)
        
        # Create updated state
        new_node_attrs = dict(state.node_attrs)
        new_node_attrs["portfolio_preferences"] = portfolio_preferences
        
        if new_tokens_spent is not None:
            new_node_attrs["tokens_spent"] = new_tokens_spent
        
        return state.replace(node_attrs=new_node_attrs)
    
    # Create prediction market transform (always first in pipeline)
    prediction_market = create_prediction_market_transform(
        config={
            "accuracy": config.market.accuracy,
            "num_options": len(config.market.initial_predictions or [1.0, 1.0, 1.0])
        }
    )
    
    # Create token refresh transform
    token_refresh = create_resource_transform(
        resource_calculator=token_budget_calculator,
        config={"refresh_period": config.token_system.refresh_period}
    )
    
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
            prediction_market,  # Always first
            token_refresh,
            voting,
            resource_transform
        )
    
    elif mechanism == "PRD":
        # Predictive Representative Democracy
        voting = create_voting_transform(
            vote_aggregator=portfolio_preference_aggregator,
            config={"mechanism_type": "representative"}
        )
        
        return sequential(
            prediction_market,  # Always first
            token_refresh,
            voting,
            resource_transform
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
            prediction_market,  # Always first
            token_refresh,
            delegation,
            power_flow,
            voting,
            resource_transform
        )
    
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")