# environments/democracy/portfolio_mechanisms.py
import jax.numpy as jnp
from typing import Dict, Any, Callable, List

from core.graph import GraphState
from core.category import Transform, sequential
from transformations.bottom_up.prediction_market import create_prediction_market_transform
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform

def create_portfolio_analyzer() -> Transform:
    """Create a transformation for analyzing portfolios"""
    def transform(state: GraphState) -> GraphState:
        # Extract state components
        token_budgets = state.node_attrs.get("token_budget")
        tokens_spent = state.node_attrs.get("tokens_spent")
        is_adversarial = state.node_attrs.get("is_adversarial")
        num_agents = state.num_nodes
        portfolios = state.global_attrs.get("portfolios", {})
        token_costs = state.global_attrs.get("token_costs", {})
        
        # Cost to analyze a portfolio
        portfolio_cost = token_costs.get("portfolio_evaluation", 20)
        basic_cost = token_costs.get("basic_participation", 20)
        
        # New portfolio preferences after analysis
        portfolio_prefs = jnp.zeros((num_agents, len(portfolios)))
        new_tokens_spent = tokens_spent.copy()
        
        # Simulate each agent's portfolio analysis based on available tokens
        for i in range(num_agents):
            # Check if agent has enough tokens for at least basic participation
            available_tokens = token_budgets[i] - tokens_spent[i]
            
            if available_tokens >= basic_cost:
                # Agents spend at least the basic participation cost
                new_tokens_spent = new_tokens_spent.at[i].add(basic_cost)
                available_tokens -= basic_cost
                
                # Different decision logic based on agent type and available tokens
                if is_adversarial[i]:
                    # Adversarial agents have opposite preferences (favor worst portfolios)
                    for j, (name, portfolio) in enumerate(portfolios.items()):
                        expected_return = portfolio["expected_return"]
                        # Invert preferences - lower returns are favored
                        if name == "Contrarian":
                            score = 9.0  # Strongly favor
                        elif name == "Conservative":
                            score = 6.0  # Moderately favor
                        elif name == "Aggressive":
                            score = 2.0  # Avoid optimal portfolio
                        else:
                            score = 5.0  # Neutral on others
                        portfolio_prefs = portfolio_prefs.at[i, j].set(score)
                else:
                    # Regular agents try to maximize returns based on information
                    for j, (name, portfolio) in enumerate(portfolios.items()):
                        expected_return = portfolio["expected_return"]
                        
                        # Scores based on expected returns and available analysis
                        if available_tokens >= portfolio_cost * len(portfolios):
                            # Thorough analysis - accurate assessment
                            if name == "Aggressive":
                                score = 9.0  # Correctly identify as optimal
                            elif name == "Market-Weighted":
                                score = 7.0  # Good but not optimal
                            elif name == "Balanced":
                                score = 6.0  # Reasonable balance
                            elif name == "Conservative":
                                score = 4.0  # Too conservative
                            else:  # Contrarian
                                score = 1.0  # Correctly identify as worst
                        else:
                            # Limited analysis - more affected by prediction market
                            if name == "Market-Weighted":
                                score = 8.0  # Seems best with limited analysis
                            elif name == "Balanced":
                                score = 7.0  # Good balance with limited info
                            elif name == "Conservative":
                                score = 5.0  # Seems safe
                            elif name == "Aggressive":
                                score = 6.0  # Risky with limited info
                            else:  # Contrarian
                                score = 3.0  # Suspicious but unsure
                        
                        portfolio_prefs = portfolio_prefs.at[i, j].set(score)
                
                # Track token spending
                analysis_tokens = min(available_tokens, portfolio_cost * len(portfolios))
                new_tokens_spent = new_tokens_spent.at[i].add(analysis_tokens)
        
        # Update state with new preferences and spent tokens
        new_node_attrs = dict(state.node_attrs)
        new_node_attrs["portfolio_preferences"] = portfolio_prefs
        new_node_attrs["tokens_spent"] = new_tokens_spent
        
        return state.replace(node_attrs=new_node_attrs)
    
    return transform

def create_portfolio_voting_transform() -> Transform:
    """Creates a voting transformation for portfolio selection"""
    def portfolio_aggregator(state: GraphState, config: Dict[str, Any]) -> jnp.ndarray:
        """Aggregates portfolio preferences into a collective decision"""
        if "portfolio_preferences" not in state.node_attrs or "voting_power" not in state.node_attrs:
            # Return a uniform distribution if preferences not available
            num_portfolios = len(state.global_attrs.get("portfolios", {}))
            return jnp.ones(num_portfolios) / num_portfolios
        
        # Get preferences and voting power
        preferences = state.node_attrs["portfolio_preferences"]
        voting_power = state.node_attrs["voting_power"]
        
        # Compute weighted average (each agent's vote weighted by voting power)
        weighted_prefs = preferences * voting_power[:, jnp.newaxis]
        
        # Sum across agents for each portfolio
        portfolio_scores = jnp.sum(weighted_prefs, axis=0)
        
        # Normalize to get a distribution
        total_score = jnp.sum(portfolio_scores)
        if total_score > 0:
            return portfolio_scores / total_score
        else:
            return jnp.ones_like(portfolio_scores) / len(portfolio_scores)
    
    return create_voting_transform(portfolio_aggregator)

def create_resource_calculator() -> Callable[[GraphState, Dict[str, Any]], float]:
    """Creates a resource calculator that applies portfolio decisions"""
    def calculator(state: GraphState, config: Dict[str, Any]) -> float:
        """Calculates resource changes based on portfolio selection"""
        # Get current decision and portfolios
        portfolios = state.global_attrs.get("portfolios", {})
        portfolio_names = list(portfolios.keys())
        
        decision = state.global_attrs.get("current_decision", 0)
        if decision >= len(portfolio_names):
            # Handle invalid decision
            return 1.0
        
        # Get the selected portfolio
        selected_name = portfolio_names[decision]
        portfolio = portfolios[selected_name]
        weights = portfolio["weights"]
        
        # Use true expected returns (actual crop yields, not prediction market)
        # In a real simulation, these would be randomly generated each round
        # For simplicity, using hard-coded values from the example
        true_returns = jnp.array([1.2, 0.5, 1.7])  # [Wheat, Corn, Fungus]
        
        # Calculate actual portfolio return
        # Weighted sum of crop returns based on portfolio allocation
        return float(jnp.sum(weights * true_returns))
    
    return calculator

def create_token_budget_refresher() -> Transform:
    """Create a transformation that refreshes token budgets periodically"""
    def transform(state: GraphState) -> GraphState:
        # Get current round and budget period info
        current_round = state.global_attrs.get("round", 0)
        budget_period = state.global_attrs.get("token_costs", {}).get("budget_period", 5)
        current_period = state.global_attrs.get("current_period", 0)
        
        # Check if we need to start a new budget period
        if current_round % budget_period == 0 and current_round > 0:
            # This is the start of a new period
            new_period = current_period + 1
            
            # Reset token spending
            new_node_attrs = dict(state.node_attrs)
            new_node_attrs["tokens_spent"] = jnp.zeros_like(state.node_attrs["tokens_spent"])
            
            # Update global attributes
            new_global_attrs = dict(state.global_attrs)
            new_global_attrs["current_period"] = new_period
            
            return state.replace(
                node_attrs=new_node_attrs,
                global_attrs=new_global_attrs
            )
        
        return state
    
    return transform

def create_portfolio_mechanism_pipeline() -> Transform:
    """Creates a transformation pipeline for portfolio-based decision making"""
    # Create individual transformations
    portfolio_analyzer = create_portfolio_analyzer()
    delegation_transform = create_delegation_transform()
    power_flow = create_power_flow_transform()
    portfolio_voting = create_portfolio_voting_transform()
    resource_transform = create_resource_transform(
        resource_calculator=create_resource_calculator(),
        config={"track_history": True}
    )
    token_refresher = create_token_budget_refresher()
    
    # Combine into mechanism-specific pipelines
    pdd_pipeline = sequential(
        token_refresher,
        portfolio_analyzer,
        portfolio_voting,
        resource_transform
    )
    
    prd_pipeline = sequential(
        token_refresher,
        portfolio_analyzer,
        portfolio_voting,  # Note: would need to be modified to only count representatives
        resource_transform
    )
    
    pld_pipeline = sequential(
        token_refresher,
        portfolio_analyzer,
        delegation_transform,
        power_flow,
        portfolio_voting,
        resource_transform
    )
    
    # Return all pipeline options
    return {
        "PDD": pdd_pipeline,
        "PRD": prd_pipeline,
        "PLD": pld_pipeline
    }