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
    
    # Shuffle all agent indices for unbiased role and adversarial assignment
    all_agent_indices = jnp.arange(num_agents)
    shuffled_indices_for_roles = jr.permutation(roles_key, all_agent_indices)
    shuffled_indices_for_adversarial = jr.permutation(adv_key, all_agent_indices)

    # 1. Determine total number of adversarial agents based on adversarial_proportion_total
    num_total_adv = int(jnp.round(num_agents * agent_settings.adversarial_proportion_total))
    is_adversarial_attr = jnp.zeros(num_agents, dtype=jnp.bool_)
    if num_total_adv > 0:
        adversarial_indices = shuffled_indices_for_adversarial[:num_total_adv]
        is_adversarial_attr = is_adversarial_attr.at[adversarial_indices].set(True)

    # 2. Assign delegate roles randomly
    is_delegate_attr = jnp.zeros(num_agents, dtype=jnp.bool_)
    if num_delegates > 0:
        delegate_indices = shuffled_indices_for_roles[:num_delegates]
        is_delegate_attr = is_delegate_attr.at[delegate_indices].set(True)

    # 3. Adjust to meet adversarial_proportion_delegates (Simplified approach)
    # This part can be complex. A simple way is to count and report.
    # A more complex way would involve swapping roles/adversarial status
    # to precisely meet the delegate adversarial quota without violating the total.
    # For now, we ensure the total adversarial count is correct.
    # The interaction with adversarial_proportion_delegates is a known complexity.
    # If precise delegate adversarial count is critical, this section needs more sophisticated logic.

    # Example: Count current adversarial delegates
    actual_adv_delegates = jnp.sum(is_adversarial_attr & is_delegate_attr)
    target_adv_delegates = int(jnp.round(num_delegates * agent_settings.adversarial_proportion_delegates))

    # (Optional) Add logging to see if the delegate proportion is naturally met or needs adjustment
    # print(f"DEBUG INIT: Target Adv Delegates: {target_adv_delegates}, Actual Initial Adv Delegates: {actual_adv_delegates}")
    # print(f"DEBUG INIT: Total Adversarial Agents: {jnp.sum(is_adversarial_attr)} (Target: {num_total_adv})")

    # Initialize cognitive resources based on the final delegate roles
    cognitive_resources_attr = jnp.where(
        is_delegate_attr,
        cognitive_resource_settings.cognitive_resources_delegate,
        cognitive_resource_settings.cognitive_resources_voter
    )

    return {
        "is_delegate": is_delegate_attr,
        "is_adversarial": is_adversarial_attr,
        "cognitive_resources": cognitive_resources_attr,  # NEW: Store cognitive resources per agent
        "tokens_spent_current_round": jnp.zeros(num_agents, dtype=jnp.int32),  # Keep for compatibility
        "voting_power": jnp.ones(num_agents, dtype=jnp.float32),
        "delegation_target": -jnp.ones(num_agents, dtype=jnp.int32),
        "cumulative_performance_score": jnp.zeros(num_agents, dtype=jnp.float32), # For historical performance
        "num_decisions_made_history": jnp.zeros(num_agents, dtype=jnp.int32),    # Count for averaging
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