# environments/democracy/initialization.py
from typing import Dict, List, Tuple, Any
import jax
import jax.numpy as jnp
import jax.random as jr

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Assuming the configuration file is now named 'configuration.py' in the same directory
from environments.democracy.configuration import PortfolioDemocracyConfig, AgentSettingsConfig, TokenBudgetConfig, CropConfig, create_thesis_baseline_config

from core.graph import GraphState # Assuming this path is correct based on project structure

# Type alias for JAX PRNG key
PRNGKey = jnp.ndarray

def initialize_agent_attributes(
    key: PRNGKey,
    num_agents: int,
    num_delegates: int,
    agent_settings: AgentSettingsConfig,
    token_settings: TokenBudgetConfig
) -> Dict[str, jnp.ndarray]:
    """
    Initializes node attributes for all agents.
    """
    attrs_key, roles_key, adv_key = jr.split(key, 3)

    # Initialize agent roles: first num_delegates are delegates, rest are voters
    is_delegate_attr = jnp.arange(num_agents) < num_delegates
    
    # Initialize token budgets based on role
    token_budget_attr = jnp.where(
        is_delegate_attr,
        token_settings.tokens_delegate_per_round,
        token_settings.tokens_voter_per_round
    )

    # Initialize adversarial status
    # Adversaries among delegates
    num_adv_delegates = int(jnp.round(num_delegates * agent_settings.adversarial_proportion_delegates))
    adv_delegate_indices = jr.choice(roles_key, jnp.arange(num_delegates), shape=(num_adv_delegates,), replace=False)
    
    is_adversarial_attr = jnp.zeros(num_agents, dtype=jnp.bool_)
    is_adversarial_attr = is_adversarial_attr.at[adv_delegate_indices].set(True)

    # Adversaries among voters
    num_total_adv = int(jnp.round(num_agents * agent_settings.adversarial_proportion_total))
    remaining_adv_needed = num_total_adv - num_adv_delegates
    
    if remaining_adv_needed > 0:
        voter_indices = jnp.arange(num_delegates, num_agents)
        adv_voter_indices = jr.choice(adv_key, voter_indices, shape=(min(remaining_adv_needed, len(voter_indices)),), replace=False)
        is_adversarial_attr = is_adversarial_attr.at[adv_voter_indices].set(True)
    elif remaining_adv_needed < 0:
        # This case means more adv delegates were assigned than total quota.
        # This could happen if adv_prop_delegates is high and adv_prop_total is low.
        # For simplicity, we'll cap at num_adv_delegates.
        # Or, could remove some adv status from delegates, but that complicates.
        # Current logic prioritizes delegate adversarial count.
        pass
    
    # Let's assume it's initialized later or is dynamic.
    # For now, a simple placeholder:
    # num_portfolios = 5 # Assuming 5 portfolios for placeholder
    # beliefs_attr = jr.uniform(belief_key, (num_agents, num_portfolios))
    # beliefs_attr = beliefs_attr / jnp.sum(beliefs_attr, axis=1, keepdims=True)

    return {
        "is_delegate": is_delegate_attr,
        "is_adversarial": is_adversarial_attr,
        "token_budget_per_round": token_budget_attr,
        "tokens_spent_current_round": jnp.zeros(num_agents, dtype=jnp.int32),
        "voting_power": jnp.ones(num_agents, dtype=jnp.float32), # Base voting power
        "delegation_target": -jnp.ones(num_agents, dtype=jnp.int32), # -1 means no delegation / votes directly
    }

def get_true_expected_yields_for_round(
    round_num: int,
    crop_configs: List[CropConfig]
) -> jnp.ndarray:
    """
    Gets the true expected yields for all crops for the current round.
    """
    yields = []
    for crop_config in crop_configs:
        # Cycle through the list of yields if round_num exceeds list length
        yield_idx = round_num % len(crop_config.true_expected_yields_per_round)
        yields.append(crop_config.true_expected_yields_per_round[yield_idx])
    return jnp.array(yields, dtype=jnp.float32)


def initialize_portfolio_democracy_graph_state(
    key: PRNGKey,
    config: PortfolioDemocracyConfig
) -> GraphState:
    """
    Initializes a complete GraphState for a portfolio democracy simulation
    based on the provided configuration.
    """
    init_key, agent_attrs_key = jr.split(key)

    # Initialize agent-specific node attributes
    node_attributes = initialize_agent_attributes(
        agent_attrs_key,
        config.num_agents,
        config.num_delegates,
        config.agent_settings,
        config.token_budget_settings
    )

    # Initialize adjacency matrices (e.g., for delegation in PLD/PRD)
    # Initially, no delegations. This matrix might represent potential rather than actual.
    adj_matrices = {
        "delegation_graph": jnp.zeros((config.num_agents, config.num_agents), dtype=jnp.float32)
    }

    # Initialize global attributes of the graph/simulation
    global_attributes = {
        "round_num": 0,
        "current_total_resources": config.resources.initial_amount,
        "resource_survival_threshold": config.resources.threshold,
        
        "crop_configs": config.crops, # Store the full crop configurations
        "portfolio_configs": config.portfolios, # Store the full portfolio configurations
        
        "current_true_expected_crop_yields": get_true_expected_yields_for_round(0, config.crops),
        "prediction_market_noise_sigma": config.market_settings.prediction_noise_sigma,
        
        "democratic_mechanism": config.mechanism,
        "simulation_seed": config.seed,

        # Token costs accessible globally
        "cost_vote": config.token_budget_settings.cost_vote,
        "cost_delegate_action": config.token_budget_settings.cost_delegate_action,

        # For storing results/history
        "resource_history": [config.resources.initial_amount],
        "decision_history": [], # List of chosen portfolio indices or similar
        "portfolio_selection_history": [], # History of chosen portfolios by round
    }

    return GraphState(
        node_attrs=node_attributes,
        adj_matrices=adj_matrices,
        global_attrs=global_attributes
    )

if __name__ == "__main__":
    # Example of initializing a graph state using a baseline config
    key = jr.PRNGKey(0)
    
    # Create a baseline PDD configuration
    pdd_baseline_config = create_thesis_baseline_config(mechanism="PDD", seed=42)
    
    print(f"Initializing GraphState for PDD baseline with {pdd_baseline_config.num_agents} agents...")
    
    initial_graph_state = initialize_portfolio_democracy_graph_state(key, pdd_baseline_config)
    
    print("\nInitialized GraphState:")
    print(f"  Number of nodes (agents): {initial_graph_state.num_nodes}")
    
    print("\n  Node Attributes (showing for first 2 agents):")
    for attr_name, attr_values in initial_graph_state.node_attrs.items():
        print(f"    {attr_name}: {attr_values[:2]}")
        
    print("\n  Global Attributes (sample):")
    print(f"    Round Number: {initial_graph_state.global_attrs['round_num']}")
    print(f"    Total Resources: {initial_graph_state.global_attrs['current_total_resources']}")
    print(f"    Mechanism: {initial_graph_state.global_attrs['democratic_mechanism']}")
    print(f"    True Expected Crop Yields (Round 0): {initial_graph_state.global_attrs['current_true_expected_crop_yields']}")
    print(f"    Number of Crop Configs: {len(initial_graph_state.global_attrs['crop_configs'])}")
    print(f"    Number of Portfolio Configs: {len(initial_graph_state.global_attrs['portfolio_configs'])}")

    # Verify adversarial assignment based on config
    num_delegates = pdd_baseline_config.num_delegates
    adv_delegates_expected = int(round(num_delegates * pdd_baseline_config.agent_settings.adversarial_proportion_delegates))
    adv_total_expected = int(round(pdd_baseline_config.num_agents * pdd_baseline_config.agent_settings.adversarial_proportion_total))
    
    actual_adv_delegates = jnp.sum(initial_graph_state.node_attrs['is_adversarial'][:num_delegates])
    actual_adv_total = jnp.sum(initial_graph_state.node_attrs['is_adversarial'])
    
    print(f"\n  Adversarial Agent Assignment:")
    print(f"    Expected Adversarial Delegates: {adv_delegates_expected}")
    print(f"    Actual Adversarial Delegates: {actual_adv_delegates}")
    print(f"    Expected Total Adversarial Agents: {adv_total_expected}")
    print(f"    Actual Total Adversarial Agents: {actual_adv_total}")

    # Check token budgets
    delegate_tokens = initial_graph_state.node_attrs['token_budget_per_round'][0] # First agent is delegate
    voter_tokens_index = num_delegates # First voter
    if voter_tokens_index < pdd_baseline_config.num_agents:
      voter_tokens = initial_graph_state.node_attrs['token_budget_per_round'][voter_tokens_index]
      print(f"    Delegate token budget: {delegate_tokens} (Expected: {pdd_baseline_config.token_budget_settings.tokens_delegate_per_round})")
      print(f"    Voter token budget: {voter_tokens} (Expected: {pdd_baseline_config.token_budget_settings.tokens_voter_per_round})")
    else:
      print(f"    Delegate token budget: {delegate_tokens} (Expected: {pdd_baseline_config.token_budget_settings.tokens_delegate_per_round})")
      print(f"    (No voters in this configuration to check specific voter token budget)")