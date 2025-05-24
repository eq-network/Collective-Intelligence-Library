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
from environments.democracy.configuration import PortfolioDemocracyConfig, AgentSettingsConfig, CropConfig,CognitiveResourceConfig, create_thesis_baseline_config

from core.graph import GraphState # Assuming this path is correct based on project structure

# Type alias for JAX PRNG key
PRNGKey = jnp.ndarray

def initialize_agent_attributes(
    key: PRNGKey,
    num_agents: int,
    num_delegates: int,
    agent_settings: AgentSettingsConfig,
    cognitive_resource_settings: CognitiveResourceConfig  # Renamed parameter
) -> Dict[str, jnp.ndarray]:
    """
    MINIMAL CHANGE: Initialize node attributes using cognitive resources instead of token budgets.
    
    ARCHITECTURAL ANALYSIS:
    - Purpose: Create initial agent attributes for simulation
    - Key Change: Replace token_budget_per_round with cognitive_resources
    - Maintain: All delegation/adversarial assignment logic unchanged
    - Preserve: Backward compatibility through parameter naming
    """
    attrs_key, roles_key, adv_key = jr.split(key, 3)

    # Initialize agent roles (UNCHANGED)
    is_delegate_attr = jnp.arange(num_agents) < num_delegates
    
    # CHANGED: Initialize cognitive resources based on role
    cognitive_resources_attr = jnp.where(
        is_delegate_attr,
        cognitive_resource_settings.cognitive_resources_delegate,  # 80 for delegates
        cognitive_resource_settings.cognitive_resources_voter      # 20 for voters
    )

    # Initialize adversarial status (UNCHANGED logic)
    num_adv_delegates = int(jnp.round(num_delegates * agent_settings.adversarial_proportion_delegates))
    adv_delegate_indices = jr.choice(roles_key, jnp.arange(num_delegates), shape=(num_adv_delegates,), replace=False)
    
    is_adversarial_attr = jnp.zeros(num_agents, dtype=jnp.bool_)
    is_adversarial_attr = is_adversarial_attr.at[adv_delegate_indices].set(True)

    num_total_adv = int(jnp.round(num_agents * agent_settings.adversarial_proportion_total))
    remaining_adv_needed = num_total_adv - num_adv_delegates
    
    if remaining_adv_needed > 0:
        voter_indices = jnp.arange(num_delegates, num_agents)
        adv_voter_indices = jr.choice(adv_key, voter_indices, shape=(min(remaining_adv_needed, len(voter_indices)),), replace=False)
        is_adversarial_attr = is_adversarial_attr.at[adv_voter_indices].set(True)

    return {
        "is_delegate": is_delegate_attr,
        "is_adversarial": is_adversarial_attr,
        "cognitive_resources": cognitive_resources_attr,  # NEW: Store cognitive resources per agent
        "tokens_spent_current_round": jnp.zeros(num_agents, dtype=jnp.int32),  # Keep for compatibility
        "voting_power": jnp.ones(num_agents, dtype=jnp.float32),
        "delegation_target": -jnp.ones(num_agents, dtype=jnp.int32),
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
    MINIMAL CHANGE: Initialize GraphState with cognitive resource settings.
    
    ARCHITECTURAL MODIFICATION:
    - Add cognitive_resource_settings to global attributes
    - Update agent initialization to use cognitive resources
    - Maintain all other initialization logic unchanged
    """
    init_key, agent_attrs_key = jr.split(key)

    # Initialize agent-specific node attributes (UPDATED)
    node_attributes = initialize_agent_attributes(
        agent_attrs_key,
        config.num_agents,
        config.num_delegates,
        config.agent_settings,
        config.cognitive_resource_settings  # Updated parameter
    )
    # Add PRD-specific node attributes here
    node_attributes["is_elected_representative"] = jnp.zeros(config.num_agents, dtype=jnp.bool_)
    node_attributes["representative_term_remaining"] = jnp.zeros(config.num_agents, dtype=jnp.int32)

    # Initialize adjacency matrices (UNCHANGED)
    adj_matrices = {
        "delegation_graph": jnp.zeros((config.num_agents, config.num_agents), dtype=jnp.float32)
    }
    # Initialize global attributes (ENHANCED)
    global_attributes = {
        "round_num": 0,
        "current_total_resources": config.resources.initial_amount,
        "resource_survival_threshold": config.resources.threshold,
        
        "crop_configs": config.crops,
        "portfolio_configs": config.portfolios,
        
        "current_true_expected_crop_yields": get_true_expected_yields_for_round(0, config.crops),
        "prediction_market_noise_sigma": config.market_settings.prediction_noise_sigma,
        
        "democratic_mechanism": config.mechanism,
        "simulation_seed": config.seed,

        # PRD Specific global attributes:
        "rounds_until_next_election_prd": 0, # Countdown to next election
        "prd_election_term_length": config.prd_election_term_length,
        # If prd_num_representatives_to_elect is None, use num_delegates
        "prd_num_representatives_to_elect": config.prd_num_representatives_to_elect if config.prd_num_representatives_to_elect is not None else config.num_delegates,

        # CHANGED: Add cognitive resource settings to global state
        "cognitive_resource_settings": config.cognitive_resource_settings,

        # Keep legacy cost attributes for compatibility
        "cost_vote": config.cognitive_resource_settings.cost_vote,
        "cost_delegate_action": config.cognitive_resource_settings.cost_delegate_action,

        # History tracking (UNCHANGED)
        "resource_history": [config.resources.initial_amount],
        "decision_history": [],
        "portfolio_selection_history": [],
    }

    return GraphState(
        node_attrs=node_attributes,
        adj_matrices=adj_matrices,
        global_attrs=global_attributes
    )