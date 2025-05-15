# For: environments/democracy/portfolio_initialization.py
# Based on input_file_12.py and requirements

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, List

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState
from environments.democracy.configuration import PortfolioDemocracyConfig

def initialize_portfolio_state(
    config: PortfolioDemocracyConfig, 
    key: jr.PRNGKey
) -> GraphState:
    """
    Initialize a JAX-compatible graph state for portfolio democracy simulations.

    Args:
        config: Configuration object for the portfolio democracy simulation.
        key: JAX PRNG key for randomization.

    Returns:
        Initialized GraphState.
    """
    num_agents = config.num_agents
    num_portfolios = len(config.get_default_strategies()) # Use get_default_strategies to ensure portfolios are loaded
    num_assets = len(config.resources.asset_names)

    key_agent_attrs, key_delegation, key_prefs, key_reps = jr.split(key, 4)

    # Node Attributes
    node_attrs = {}

    # Token Budgets and Spending
    # Assign token budgets based on capacity distribution
    # For simplicity, let's distribute capacities roughly. A more complex setup would read precise distributions.
    agent_capacities = []
    low_c = config.token_system.capacity_distribution.get("low_count", num_agents // 3)
    high_c = config.token_system.capacity_distribution.get("high_count", num_agents // 3)
    med_c = num_agents - low_c - high_c
    
    capacities_map = config.token_system.capacity_levels
    
    for _ in range(low_c): agent_capacities.append(capacities_map.get("low", 150))
    for _ in range(med_c): agent_capacities.append(capacities_map.get("medium", 300))
    for _ in range(high_c): agent_capacities.append(capacities_map.get("high", 500))
    
    # Shuffle if needed, or assign sequentially for now
    node_attrs["token_budget"] = jnp.array(agent_capacities[:num_agents], dtype=jnp.float32)
    node_attrs["tokens_spent"] = jnp.zeros(num_agents, dtype=jnp.float32)

    # Adversarial Agents
    num_adversarial = int(num_agents * config.agents.adversarial_proportion)
    adversarial_flags = jnp.zeros(num_agents, dtype=jnp.bool_)
    if config.agents.adversarial_introduction == "immediate" and num_adversarial > 0:
        adversarial_indices = jr.choice(key_agent_attrs, jnp.arange(num_agents), shape=(num_adversarial,), replace=False)
        adversarial_flags = adversarial_flags.at[adversarial_indices].set(True)
    node_attrs["is_adversarial"] = adversarial_flags

    # Representatives (for PRD)
    # Placeholder: actual representative selection would be more complex or configured
    is_representative = jnp.zeros(num_agents, dtype=jnp.bool_)
    if config.mechanism == "PRD":
        # Example: select first N/10 agents as representatives
        num_representatives = max(1, num_agents // 10) 
        rep_indices = jr.choice(key_reps, jnp.arange(num_agents), shape=(num_representatives,), replace=False)
        is_representative = is_representative.at[rep_indices].set(True)
    node_attrs["is_representative"] = is_representative
    
    # Portfolio Preferences (initial, e.g., uniform or slightly noisy)
    initial_preferences = jr.uniform(key_prefs, (num_agents, num_portfolios))
    node_attrs["portfolio_preferences"] = initial_preferences / jnp.sum(initial_preferences, axis=1, keepdims=True)

    # Voting Power (initial, typically uniform)
    node_attrs["voting_power"] = jnp.ones(num_agents, dtype=jnp.float32)
    
    # Expertise (if used by analysis adapter) - example Beta distribution
    expertise_alpha = config.agents.expertise_distribution.get("alpha", 2.0)
    expertise_beta = config.agents.expertise_distribution.get("beta", 5.0)
    node_attrs["expertise"] = jr.beta(key_agent_attrs, expertise_alpha, expertise_beta, (num_agents,))

    # Delegation Choices (for PLD, initial: no delegation or random)
    # -1 or self-index could mean no delegation / direct voting
    # For now, initialize to no delegation (represented by delegating to oneself, to be filtered by delegation_transform)
    delegation_choices = jr.randint(key_delegation, (num_agents,), 0, num_agents) 
    # A more common initial state is no delegation (e.g., all -1, or handled by voting_power if no delegation matrix)
    # Let's set to -1 to indicate direct voting initially. The delegation_transform should handle this.
    # Or, if delegation matrix starts empty, power_flow will start with direct power.
    # For simplicity, PLD usually starts with agents making explicit choices or a pre-defined network.
    # If `delegation_choices` is used by `create_delegation_transform`, it should be ints.
    node_attrs["delegation_choices"] = jnp.array([-1]*num_agents, dtype=jnp.int32) # -1 for no delegation initially

    # Adjacency Matrices
    adj_matrices = {}
    # Example: initial communication network (e.g., fully connected for simplicity, or from config)
    # adj_matrices["communication"] = jnp.ones((num_agents, num_agents)) - jnp.eye(num_agents)
    # For PLD, an initial delegation graph might be set up here or be empty
    adj_matrices["delegation"] = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)


    # Global Attributes
    global_attrs = {}
    global_attrs["portfolios"] = {
        name: {"weights": jnp.array(strategy.weights), 
               "description": strategy.description,
               "risk_level": strategy.risk_level,
               "metadata": strategy.metadata,
               # Store expected return if needed for rule-based, though LLM might infer
               "expected_return": jnp.sum(jnp.array(strategy.weights) * jnp.array(config.market.initial_predictions))
              } 
        for name, strategy in config.get_default_strategies().items()
    }
    global_attrs["asset_names"] = config.resources.asset_names
    global_attrs["num_assets"] = len(config.resources.asset_names)
    global_attrs["prediction_market"] = jnp.array(config.market.initial_predictions, dtype=jnp.float32) # Predictions for assets
    global_attrs["true_returns"] = jnp.array(config.market.true_returns, dtype=jnp.float32) # Actual returns for assets for this round
    
    global_attrs["token_costs"] = config.token_system.operation_costs
    global_attrs["token_refresh_period"] = config.token_system.refresh_period

    global_attrs["total_resources"] = jnp.array(config.resources.initial_amount, dtype=jnp.float32)
    global_attrs["resource_threshold"] = jnp.array(config.resources.threshold, dtype=jnp.float32)
    global_attrs["round"] = 0
    global_attrs["current_period"] = 0 # For token budget refresh
    global_attrs["trade_history"] = [] # If market mechanics are added
    global_attrs["resource_history"] = [] # For tracking resource evolution

    return GraphState(node_attrs=node_attrs, adj_matrices=adj_matrices, global_attrs=global_attrs)

if __name__ == '__main__':
    # Example Usage:
    default_config = PortfolioDemocracyConfig() # Uses create_default_portfolio_config defaults
    
    # More specific config
    custom_config = PortfolioDemocracyConfig(
        mechanism="PLD",
        num_agents=20,
        num_rounds=10,
        seed=123,
        resources=default_config.resources.replace(initial_amount=200.0, asset_names=["StockA", "StockB", "BondA"]),
        token_system=default_config.token_system.replace(capacity_levels={"low":50, "medium":100, "high":150}),
        market=default_config.market.replace(initial_predictions=[1.05, 0.98, 1.01], true_returns=[1.07, 0.95, 1.00]),
        agents=default_config.agents.replace(adversarial_proportion=0.1),
        # strategies will use defaults if not overridden
    )
    
    prng_key = jr.PRNGKey(custom_config.seed)
    initial_gs = initialize_portfolio_state(custom_config, prng_key)

    print(f"Initialized GraphState for {custom_config.num_agents} agents.")
    print("Node Attributes Keys:", list(initial_gs.node_attrs.keys()))
    print("Token Budgets Sample:", initial_gs.node_attrs["token_budget"][:5])
    print("Adversarial Flags Sample:", initial_gs.node_attrs["is_adversarial"][:5])
    print("Global Attributes Keys:", list(initial_gs.global_attrs.keys()))
    print("Portfolios:", {name: data["weights"] for name, data in initial_gs.global_attrs["portfolios"].items()})
    print(f"Total Resources: {initial_gs.global_attrs['total_resources']}")
    print(f"Initial Prediction Market: {initial_gs.global_attrs['prediction_market']}")
    print(f"Token costs: {initial_gs.global_attrs['token_costs']}")
    # Check if portfolio strategies were populated
    assert len(initial_gs.global_attrs["portfolios"]) > 0, "Portfolios were not initialized."
    print(f"Number of portfolios: {len(initial_gs.global_attrs['portfolios'])}")
    assert len(initial_gs.node_attrs["portfolio_preferences"][0]) == len(initial_gs.global_attrs["portfolios"]), "Prefs shape mismatch"
    
    # Test that get_default_strategies is working within the config
    strat_check_config = PortfolioDemocracyConfig()
    initialized_strategies = strat_check_config.get_default_strategies()
    assert len(initialized_strategies) > 0, "get_default_strategies() returned empty."
    print("Default strategies loaded:", list(initialized_strategies.keys()))

    # Test with overridden strategies
    custom_strategies = {
        "CustomStrat1": default_config.strategies.get("Conservative").replace(name="CustomStrat1", weights=[0.8,0.1,0.1])
    }
    config_with_custom_strats = PortfolioDemocracyConfig(strategies=custom_strategies)
    # Re-initialize to test this path
    gs_custom_strats = initialize_portfolio_state(config_with_custom_strats, prng_key)
    assert "CustomStrat1" in gs_custom_strats.global_attrs["portfolios"]
    assert "Conservative" not in gs_custom_strats.global_attrs["portfolios"] # Default should be overridden
    print("Successfully initialized with custom strategies.")

# Replace the content of environments/democracy/portfolio_config.py if needed
# For this example, I'll assume input_file_6.py is already correct.
# The main modification here is to ensure initialize_portfolio_state correctly
# uses config.get_default_strategies() to populate portfolios if config.strategies is empty.
# The provided portfolio_config.py (input_file_6) `__post_init__` cannot modify self.strategies
# because it's frozen. The `get_default_strategies` method is a workaround.
# `initialize_portfolio_state` is the consumer that can apply this logic.
print("portfolio_initialization.py content generated.")
