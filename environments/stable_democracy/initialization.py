# environments/stable_democracy/initialization.py
from typing import Dict, List, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jr

import sys
from pathlib import Path

# Assuming this file is in environments/stable_democracy
# and core is at project_root/core
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.graph import GraphState
# Import from this package's configuration
from .configuration import (
    StablePortfolioDemocracyConfig, AgentSettingsConfig, CropConfig, 
    ParticipationConfig, LockedValueConfig
    # If CognitiveResourceConfig is needed and defined elsewhere (e.g. noise_democracy), import it
    # from environments.noise_democracy.configuration import CognitiveResourceConfig
)
# For this example, we assume CognitiveResourceConfig is NOT the primary driver if part. system is on.
# If it's used for prediction noise with locked values, it needs to be available.
# Let's assume it's an optional part of StablePortfolioDemocracyConfig for now.


PRNGKey = jnp.ndarray

def initialize_stable_agent_attributes(
    key: PRNGKey,
    num_agents: int,
    num_delegates: int,
    agent_settings: AgentSettingsConfig,
    participation_settings: ParticipationConfig, # Required for stable_democracy
    # cognitive_resource_settings: Optional[CognitiveResourceConfig] = None # If used for prediction noise
) -> Dict[str, jnp.ndarray]:
    """
    Initialize node attributes for the stable democracy system.
    Focuses on participation attributes. Cognitive resources are secondary.
    """
    attrs_key, roles_key, adv_key = jr.split(key, 3)
    
    all_agent_indices = jnp.arange(num_agents)
    shuffled_indices_for_roles = jr.permutation(roles_key, all_agent_indices)
    shuffled_indices_for_adversarial = jr.permutation(adv_key, all_agent_indices)

    num_total_adv = int(jnp.round(num_agents * agent_settings.adversarial_proportion_total))
    is_adversarial_attr = jnp.zeros(num_agents, dtype=jnp.bool_)
    if num_total_adv > 0:
        adversarial_indices = shuffled_indices_for_adversarial[:num_total_adv]
        is_adversarial_attr = is_adversarial_attr.at[adversarial_indices].set(True)

    is_delegate_attr = jnp.zeros(num_agents, dtype=jnp.bool_)
    if num_delegates > 0:
        delegate_indices = shuffled_indices_for_roles[:num_delegates]
        is_delegate_attr = is_delegate_attr.at[delegate_indices].set(True)

    attributes = {
        "is_delegate": is_delegate_attr,
        "is_adversarial": is_adversarial_attr,
        "voting_power": jnp.ones(num_agents, dtype=jnp.float32), # Base voting power
        "delegation_target": -jnp.ones(num_agents, dtype=jnp.int32), # No delegation initially
        
        # Participation attributes
        "can_participate_this_round": jnp.ones(num_agents, dtype=jnp.bool_), # All participate in round 0
        "participated_last_round": jnp.zeros(num_agents, dtype=jnp.bool_), # None participated before sim start
        "participation_history": jnp.zeros(num_agents, dtype=jnp.int32), # Cumulative count
        "forced_delegation_this_round": jnp.zeros(num_agents, dtype=jnp.bool_),

        # Attributes for compatibility or potential advanced features (may not be heavily used in pure stable system)
        "tokens_spent_current_round": jnp.zeros(num_agents, dtype=jnp.int32), # For compatibility with some transforms
        "cumulative_performance_score": jnp.zeros(num_agents, dtype=jnp.float32),
        "num_decisions_made_history": jnp.zeros(num_agents, dtype=jnp.int32),
        # "cognitive_resources" might be added if config.cognitive_resource_settings is not None
        # For now, let's assume it's not the primary attribute for this system.
        # If it IS needed for prediction noise (when eliminate_prediction_noise=False), it should be initialized.
        # This depends on how StablePortfolioDemocracyConfig handles optional cog_config.
        # If config.cognitive_resource_settings is present:
        # cog_res_vals = jnp.where(is_delegate_attr, 
        #                          config.cognitive_resource_settings.cognitive_resources_delegate, 
        #                          config.cognitive_resource_settings.cognitive_resources_voter)
        # attributes["cognitive_resources"] = cog_res_vals
    }
    return attributes

def get_stable_true_expected_yields_for_round(
    round_num: int, # May not be used if yields are truly static per crop
    crop_configs: List[CropConfig],
    locked_value_config: Optional[LockedValueConfig] = None
) -> jnp.ndarray:
    """
    Gets true expected yields. If locked_value_config is active and specifies,
    it returns the locked yields. Otherwise, it uses dynamic yields from CropConfig.
    """
    if locked_value_config and locked_value_config.use_locked_values:
        # Use locked_crop_yields. These are per-crop, constant across rounds.
        locked_yields_list = locked_value_config.locked_crop_yields
        if not crop_configs: return jnp.array([], dtype=jnp.float32)
        num_defined_crops = len(crop_configs)

        if not locked_yields_list: # No locked yields provided, despite use_locked_values=True
            print(f"Warning (stable_init): use_locked_values is True, but locked_crop_yields is empty. Defaulting to 1.0 for {num_defined_crops} crops.")
            return jnp.ones(num_defined_crops, dtype=jnp.float32)

        # Ensure we return an array of length num_defined_crops
        if len(locked_yields_list) == num_defined_crops:
            return jnp.array(locked_yields_list, dtype=jnp.float32)
        else: # Mismatch, cycle or truncate locked_yields_list
            print(f"Warning (stable_init): Mismatch num_crops ({num_defined_crops}) and locked_yields ({len(locked_yields_list)}). Cycling/truncating.")
            return jnp.array([locked_yields_list[i % len(locked_yields_list)] for i in range(num_defined_crops)], dtype=jnp.float32)

    # Fallback to dynamic yields from CropConfig if not using locked values
    if not crop_configs: return jnp.array([], dtype=jnp.float32)
    yields = []
    for crop_config in crop_configs:
        if not crop_config.true_expected_yields_per_round:
             yields.append(1.0) # Default
             continue
        yield_idx = round_num % len(crop_config.true_expected_yields_per_round)
        yields.append(crop_config.true_expected_yields_per_round[yield_idx])
    return jnp.array(yields, dtype=jnp.float32)


def initialize_stable_democracy_graph_state(
    key: PRNGKey,
    config: StablePortfolioDemocracyConfig # Uses the new config type
) -> GraphState:
    """
    Initialize GraphState for the stable democracy system.
    """
    init_key, agent_attrs_key = jr.split(key)

    node_attributes = initialize_stable_agent_attributes(
        agent_attrs_key,
        config.num_agents,
        config.num_delegates,
        config.agent_settings,
        config.participation_settings, # Pass participation_settings
        # config.cognitive_resource_settings # Pass if it's part of StablePortfolioDemocracyConfig and used
    )
    # Add cognitive_resources if the config has it (for scenarios with noisy perception of locked values)
    if config.cognitive_resource_settings:
        is_delegate_attr = node_attributes["is_delegate"]
        node_attributes["cognitive_resources"] = jnp.where(
            is_delegate_attr,
            config.cognitive_resource_settings.cognitive_resources_delegate,
            config.cognitive_resource_settings.cognitive_resources_voter
        )

    # Initialize last_portfolio_vote (num_agents, num_portfolios)
    num_portfolios = len(config.portfolios) if config.portfolios else 0
    node_attributes["last_portfolio_vote"] = jnp.zeros((config.num_agents, num_portfolios), dtype=jnp.int32)


    # PRD-specific node attributes
    node_attributes["is_elected_representative"] = jnp.zeros(config.num_agents, dtype=jnp.bool_)
    node_attributes["representative_term_remaining"] = jnp.zeros(config.num_agents, dtype=jnp.int32)

    adj_matrices = {
        "delegation_graph": jnp.zeros((config.num_agents, config.num_agents), dtype=jnp.float32)
    }
    
    global_attributes = {
        "round_num": -1, 
        "current_total_resources": config.resources.initial_amount,
        "resource_survival_threshold": config.resources.threshold,
        
        "crop_configs": config.crops, # Still useful for names, structure, even if yields are locked
        "portfolio_configs": config.portfolios,
        
        "current_true_expected_crop_yields": get_stable_true_expected_yields_for_round(
            0, config.crops, config.locked_value_settings
        ),
        # prediction_market_noise_sigma is from a MarketConfig, which StablePortfolioDemocracyConfig doesn't directly hold.
        # If needed (i.e., eliminate_prediction_noise=False), it should be part of LockedValueConfig or a new MarketConfig within Stable...
        # For now, assume if eliminate_prediction_noise=False, a default sigma is used or passed via config.
        "prediction_market_noise_sigma": 0.25 if (config.locked_value_settings and not config.locked_value_settings.eliminate_prediction_noise) else 0.0,

        "democratic_mechanism": config.mechanism,
        "simulation_seed": config.seed,

        "rounds_until_next_election_prd": 0,
        "prd_election_term_length": config.prd_election_term_length,
        "prd_num_representatives_to_elect": config.prd_num_representatives_to_elect,

        "participation_settings": config.participation_settings,
        "locked_value_settings": config.locked_value_settings,
        "cognitive_resource_settings": config.cognitive_resource_settings, # Store if present

        "resource_history": [config.resources.initial_amount],
        "decision_history": [],
        "portfolio_selection_history": [],
    }

    return GraphState(
        node_attrs=node_attributes,
        adj_matrices=adj_matrices,
        global_attrs=global_attributes
    )