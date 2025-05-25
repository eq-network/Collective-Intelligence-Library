# transformations/bottom_up/prediction_market.py
"""
Prediction market transformation - a general mechanism for incorporating
prediction signals into the graph state.

This transform follows the inversion of control principle, accepting a prediction
generator function rather than embedding domain-specific prediction logic.
"""
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Callable

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState

# Update the existing prediction market transform
def create_prediction_market_transform(
    prediction_generator: Callable[[GraphState, Dict[str, Any]], jnp.ndarray] = None,
    config: Dict[str, Any] = None
) -> Callable[[GraphState], GraphState]:
    """
    MINIMAL CHANGE: Updated to handle agent-specific signals.
    
    BACKWARD COMPATIBILITY: If no cognitive resources config present, 
    falls back to original uniform signal behavior.
    """
    config = config or {}
    
    def default_generator(state: GraphState, config: Dict[str, Any]) -> jnp.ndarray:
        return state.global_attrs.get("current_true_expected_crop_yields", jnp.ones(3))
    
    generator = prediction_generator or default_generator
    
    def transform(state: GraphState) -> GraphState:
        # Check if we have cognitive resources configuration
        cognitive_config = state.global_attrs.get("cognitive_resource_settings")
        
        if cognitive_config is None:
            # BACKWARD COMPATIBILITY: Use original uniform signal generation
            true_yields = generator(state, config)
            key_val = state.global_attrs.get("round_num", 0) + state.global_attrs.get("simulation_seed", 0)
            key = jr.PRNGKey(key_val)
            noise_sigma = state.global_attrs.get("prediction_market_noise_sigma", 0.25)
            noise = jr.normal(key, shape=true_yields.shape) * noise_sigma
            uniform_signals = true_yields + noise
            
            new_global_attrs = dict(state.global_attrs)
            new_global_attrs["prediction_market_crop_signals"] = uniform_signals
            return state.replace(global_attrs=new_global_attrs)
        
        # NEW BEHAVIOR: Generate agent-specific signals
        true_yields = generator(state, config)
        
        # Update state to include cognitive config for signal generator
        temp_global_attrs = dict(state.global_attrs)
        temp_global_attrs["cognitive_resource_settings"] = cognitive_config
        temp_state = state.replace(global_attrs=temp_global_attrs)
        
        # Generate signals using enhanced generator
        signal_results = _enhanced_prediction_market_signal_generator(temp_state, config)
        
        # Update global attributes
        new_global_attrs = dict(state.global_attrs)
        new_global_attrs["agent_specific_prediction_signals"] = signal_results["agent_specific_signals"]
        new_global_attrs["prediction_market_crop_signals"] = signal_results["market_consensus"]  # Backward compatibility
        
        return state.replace(global_attrs=new_global_attrs)
    
    return transform


def _enhanced_prediction_market_signal_generator(state: GraphState, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced signal generator that creates agent-specific signals based on cognitive resources.
    
    MATHEMATICAL MODEL:
    - Base noise: σ_base (same for all agents)
    - Additional noise: (100 - cognitive_resources) / 100  
    - Total noise: σ_base + additional_noise
    
    EXAMPLE:
    - Base noise = 0.25
    - Delegate (80 resources): total_noise = 0.25 + (100-80)/100 = 0.25 + 0.2 = 0.45
    - Voter (20 resources): total_noise = 0.25 + (100-20)/100 = 0.25 + 0.8 = 1.05
    """
    key_val = state.global_attrs.get("round_num", 0) + state.global_attrs.get("simulation_seed", 0)
    base_key = jr.PRNGKey(key_val)
    
    true_expected_yields = state.global_attrs["current_true_expected_crop_yields"]
    base_noise_sigma = state.global_attrs["prediction_market_noise_sigma"]
    
    # Get cognitive resource settings
    cognitive_config = state.global_attrs.get("cognitive_resource_settings")
    if cognitive_config is None:
        # Fallback to uniform signals (backward compatibility)
        noise = jr.normal(base_key, shape=true_expected_yields.shape) * base_noise_sigma
        uniform_signals = true_expected_yields + noise
        return {"uniform_signals": uniform_signals}
    
    # Generate agent-specific signals
    num_agents = state.num_nodes
    agent_signals = {}
    
    for agent_id in range(num_agents):
        # Get agent's cognitive resources
        is_delegate = state.node_attrs["is_delegate"][agent_id]
        if is_delegate:
            cognitive_resources = cognitive_config.cognitive_resources_delegate
        else:
            cognitive_resources = cognitive_config.cognitive_resources_voter
        
        # Calculate total noise for this agent
        additional_noise = (100 - cognitive_resources) / 100.0
        total_noise_sigma = base_noise_sigma + additional_noise
        
        # Generate agent-specific key and signals
        agent_key = jr.split(base_key, num_agents + 1)[agent_id]
        noise = jr.normal(agent_key, shape=true_expected_yields.shape) * total_noise_sigma
        agent_prediction_signals = true_expected_yields + noise
        
        agent_signals[agent_id] = agent_prediction_signals
    
    # CLEAN DEBUG: Simple round-level information only
    round_num = state.global_attrs.get("round_num", 0)
    print(f"[R{round_num:02d}] Prediction signals generated for {num_agents} agents")
    
    # Return both agent-specific signals and a consensus signal
    best_agent_id = 0  # Delegate with highest cognitive resources
    for agent_id in range(num_agents):
        if state.node_attrs["is_delegate"][agent_id]:
            best_agent_id = agent_id
            break
    
    return {
        "agent_specific_signals": agent_signals,
        "market_consensus": agent_signals[best_agent_id]
    }