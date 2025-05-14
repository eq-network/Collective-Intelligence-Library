# environments/democracy/portfolio_initialization.py
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, List, Any
from core.graph import GraphState
from environments.democracy.portfolio_config import PortfolioDemocracyConfig

def initialize_token_budgets(
    num_agents: int,
    config: TokenBudgetConfig,
    key: jr.PRNGKey
) -> jnp.ndarray:
    """Initialize token budgets for all agents"""
    token_budgets = jnp.zeros(num_agents)
    
    # Assign agent capacity types
    agent_indices = jnp.arange(num_agents)
    key, subkey = jr.split(key)
    agent_indices = jr.permutation(subkey, agent_indices)
    
    # Assign token budgets based on agent capacity
    low_idx = agent_indices[:config.low_capacity_agents]
    med_idx = agent_indices[config.low_capacity_agents:
                           config.low_capacity_agents + config.medium_capacity_agents]
    high_idx = agent_indices[config.low_capacity_agents + config.medium_capacity_agents:
                            config.low_capacity_agents + config.medium_capacity_agents + 
                            config.high_capacity_agents]
    
    token_budgets = token_budgets.at[low_idx].set(config.low_budget)
    token_budgets = token_budgets.at[med_idx].set(config.medium_budget)
    token_budgets = token_budgets.at[high_idx].set(config.high_budget)
    
    return token_budgets

def generate_portfolios(
    crops: List[CropConfig],
    num_portfolios: int,
    portfolio_types: List[str],
    prediction_market: jnp.ndarray,
    key: jr.PRNGKey
) -> Dict[str, Any]:
    """Generate different portfolio allocations"""
    num_crops = len(crops)
    crop_names = [crop.name for crop in crops]
    
    # Create different portfolio strategies
    portfolios = {}
    
    # Example portfolio generation (simplified)
    # Conservative: Weighted toward lower-variance crops
    # Balanced: Relatively even distribution
    # Aggressive: Weighted toward predicted high-return crops
    # Contrarian: Opposite of market predictions
    # Random: Random weights
    
    # Create actual portfolio definitions based on crop and market information
    portfolios["Conservative"] = {"weights": jnp.array([0.6, 0.3, 0.1])}
    portfolios["Balanced"] = {"weights": jnp.array([0.4, 0.3, 0.3])}
    portfolios["Aggressive"] = {"weights": jnp.array([0.2, 0.1, 0.7])}
    portfolios["Contrarian"] = {"weights": jnp.array([0.3, 0.6, 0.1])}
    portfolios["Market-Weighted"] = {"weights": jnp.array([0.35, 0.15, 0.5])}
    
    # Calculate expected returns based on market predictions
    for name, portfolio in portfolios.items():
        weights = portfolio["weights"]
        expected_return = jnp.sum(weights * prediction_market)
        portfolios[name]["expected_return"] = float(expected_return)
    
    return portfolios

def initialize_portfolio_state(config: PortfolioDemocracyConfig, key: jr.PRNGKey) -> GraphState:
    """Initialize graph state for portfolio voting simulation"""
    num_agents = config.num_agents
    num_crops = len(config.crops)
    crop_names = [crop.name for crop in config.crops]
    
    # Split key for various initialization needs
    key, subkey1, subkey2, subkey3, subkey4, subkey5 = jr.split(key, 6)
    
    # Initialize token budgets
    token_budgets = initialize_token_budgets(num_agents, config.token_budget, subkey1)
    
    # Initialize agent attributes
    expertise = jr.beta(subkey2, 2.0, 3.0, (num_agents,))
    
    # Initialize adversarial flags (some agents act against group interest)
    is_adversarial = jnp.zeros((num_agents,), dtype=bool)
    if config.adversarial.proportion > 0:
        num_adversarial = int(num_agents * config.adversarial.proportion)
        indices = jr.choice(
            subkey3,
            jnp.arange(num_agents),
            shape=(num_adversarial,),
            replace=False
        )
        is_adversarial = is_adversarial.at[indices].set(True)
    
    # Initialize communication network
    communication = jnp.zeros((num_agents, num_agents))
    # For simplicity, using a basic random network - this could be more sophisticated
    connection_prob = 0.2
    random_connections = jr.uniform(subkey4, (num_agents, num_agents)) < connection_prob
    # Make connections symmetric
    random_connections = jnp.logical_or(random_connections, random_connections.T)
    # No self-connections
    communication = jnp.where(
        jnp.logical_and(random_connections, ~jnp.eye(num_agents, dtype=bool)),
        1.0, 0.0
    )
    
    # Initialize prediction market signals
    # This is a simplified prediction market
    prediction_market = jnp.array([1.15, 0.6, 1.5])  # Example from scenario
    
    # Generate portfolios based on market signals
    portfolios = generate_portfolios(
        config.crops,
        config.portfolios.num_portfolios,
        config.portfolios.portfolio_types,
        prediction_market,
        subkey5
    )
    
    # Create initial voting preferences (will be updated by agents)
    portfolio_preferences = jnp.zeros((num_agents, len(portfolios)))
    
    # Combine into graph state
    node_attrs = {
        "expertise": expertise,
        "is_adversarial": is_adversarial,
        "token_budget": token_budgets,
        "tokens_spent": jnp.zeros(num_agents),
        "portfolio_preferences": portfolio_preferences,
        "voting_power": jnp.ones(num_agents),  # Each agent starts with 1 vote
    }
    
    adj_matrices = {
        "communication": communication,
        "delegation": jnp.zeros((num_agents, num_agents)),  # No initial delegations
    }
    
    global_attrs = {
        "round": 0,
        "total_resources": config.resources.initial_amount,
        "resource_min_threshold": config.resources.threshold,
        "crops": crop_names,
        "portfolios": portfolios,
        "prediction_market": prediction_market,
        "token_costs": config.token_budget.token_costs,
        "current_period": 0,  # Track budget periods
    }
    
    return GraphState(node_attrs, adj_matrices, global_attrs)