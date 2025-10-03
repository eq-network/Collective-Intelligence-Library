# environments/democracy/stable/agent_interface.py
"""
This module provides the transformation that bridges the pure GraphState world
with the object-oriented Agent world. It calls the `act` method on agent objects.
"""
from typing import List, Optional, Dict, Any
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor

from core.category import Transform
from core.graph import GraphState
from core.agents import Agent
from services.llm import LLMService

def create_llm_decision_transform(
    agents: List[Agent],
    llm_service: Optional[LLMService],
    prompt_logic_type: str = "default" # For multi-phase mechs like PLD
) -> Transform:
    """
    Creates a transform that gets actions from agent objects and updates the graph state.
    """
    def transform(state: GraphState) -> GraphState:
        participating_agent_indices = jnp.where(state.node_attrs['can_participate_this_round'])[0]
        
        # Prepare observations for all participating agents
        observations = {}
        for agent_id in participating_agent_indices:
            # Construct the observation dict for this agent from the current state
            agent_signals = state.global_attrs['agent_specific_prediction_signals'][agent_id]
            # ... format portfolio_options_str, etc. ...
            observations[agent_id] = {
                "round_num": state.global_attrs['round_num'],
                "mechanism": state.global_attrs['config_ref'].mechanism,
                "portfolio_options_str": "...", # Formatted string
                "portfolio_yields": agent_signals, # Raw signals for hardcoded agents
                "prompt_logic_type": prompt_logic_type,
                # ... add other necessary info ...
            }
            
        # Get actions from agents in parallel
        actions = {}
        with ThreadPoolExecutor(max_workers=len(participating_agent_indices)) as executor:
            future_to_agent_id = {
                executor.submit(agents[i].act, observations[i]): i
                for i in participating_agent_indices
            }
            for future in as_completed(future_to_agent_id):
                agent_id = future_to_agent_id[future]
                actions[agent_id] = future.result()

        # Update the graph state with the results of the actions
        new_node_attrs = state.node_attrs.copy()
        for agent_id, action in actions.items():
            if action.get("type") == "vote":
                new_node_attrs['agent_portfolio_votes'] = \
                    new_node_attrs['agent_portfolio_votes'].at[agent_id].set(jnp.array(action['votes']))
            elif action.get("type") == "delegate":
                new_node_attrs['delegation_target'] = \
                    new_node_attrs['delegation_target'].at[agent_id].set(action['target_id'])
        
        return state.replace(node_attrs=new_node_attrs)
        
    return transform