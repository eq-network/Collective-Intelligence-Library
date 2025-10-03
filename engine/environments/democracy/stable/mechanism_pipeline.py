# environments/democracy/stable/mechanism_pipeline.py
"""
The "Master Composer" for the Stable Democracy Simulation Environment.

This module's sole responsibility is to assemble and compose sequences of
pure graph transformations to create a single, complete round transformation
function (`F` in the architectural diagram) for a given mechanism.
"""
from typing import Optional, Dict, List
import jax.numpy as jnp

from core.category import Transform, sequential
from core.graph import GraphState
from services.llm import LLMService

from .configuration import StablePortfolioDemocracyConfig
from .initialization import get_stable_true_expected_yields_for_round
from engine.environments.democracy.helper.voting_aggregation import _portfolio_vote_aggregator_stable
from .agent_interface import create_llm_decision_transform # We will create this interface file

from engine.environments.democracy.transforms.participation import create_participation_constraint_transform
from engine.environments.democracy.transforms.delegation import create_delegation_transform
from engine.environments.democracy.transforms.power_flow import create_power_flow_transform
from engine.environments.democracy.transforms.voting import create_voting_transform
from engine.environments.democracy.transforms.election import create_election_transform
from transformations.top_down.resource import create_resource_transform

# --- Phase-Specific Transformations ---

def _create_housekeeping_and_signals_transform() -> Transform:
    """Composes the initial steps of any round: updating round state and providing perfect info."""
    def housekeeping_transform(state: GraphState) -> GraphState:
        new_globals = state.global_attrs.copy()
        new_globals["round_num"] = state.global_attrs.get("round_num", -1) + 1
        config = state.global_attrs['config_ref']
        
        true_yields = get_stable_true_expected_yields_for_round(
            new_globals["round_num"],
            config.crops,
            None # LockedValueConfig is simplified for this example
        )
        new_globals["current_true_expected_crop_yields"] = true_yields
        new_globals["current_actual_crop_yields"] = true_yields
        return state.replace(global_attrs=new_globals)
        
    def prediction_signal_transform(state: GraphState) -> GraphState:
        true_yields = state.global_attrs.get("current_true_expected_crop_yields", jnp.array([]))
        agent_signals = {i: true_yields for i in range(state.num_nodes)}
        new_globals = state.global_attrs.copy()
        new_globals["agent_specific_prediction_signals"] = agent_signals
        return state.replace(global_attrs=new_globals)
        
    return sequential(housekeeping_transform, create_participation_constraint_transform(), prediction_signal_transform)

def _create_resource_allocation_transform(config: StablePortfolioDemocracyConfig) -> Transform:
    """Creates the transform for the final resource allocation step."""
    def _resource_calculator(state: GraphState, _: Dict) -> float:
        decision_idx = state.global_attrs.get("current_decision", -1)
        if 0 <= decision_idx < len(config.portfolios):
            return config.portfolios[decision_idx].true_expected_yield
        return 1.0

    return create_resource_transform(
        resource_calculator=_resource_calculator,
        config={"resource_attr_name": "current_total_resources"}
    )

# --- The Main Pipeline Factory ---

def create_stable_democracy_pipeline(
    config: StablePortfolioDemocracyConfig,
    agents: List[Any], # List of agent objects
    llm_service: Optional[LLMService]
) -> Transform:
    """
    Assembles the correct pipeline of transformations for the specified mechanism.
    This is a direct implementation of the architectural diagram.
    """
    
    # Universal starting and ending transforms
    housekeeping_and_signals = _create_housekeeping_and_signals_transform()
    resource_allocation = _create_resource_allocation_transform(config)
    
    # The core decision transform, now interfacing with agent objects
    llm_decision_transform = create_llm_decision_transform(agents, llm_service)
    
    # Final voting transform, which uses the aggregator function
    final_voting_transform = create_voting_transform(
        vote_aggregator=_portfolio_vote_aggregator_stable,
        config={"mechanism_type": config.mechanism}
    )

    # --- Assemble the pipeline based on the mechanism ---
    if config.mechanism == 'PDD':
        return sequential(
            housekeeping_and_signals,
            llm_decision_transform,      # Agents decide votes
            final_voting_transform,      # Votes are aggregated
            resource_allocation          # Resources are updated
        )

    elif config.mechanism == 'PRD':
        election_transform = create_election_transform()
        return sequential(
            housekeeping_and_signals,
            election_transform,          # Checks if election happens, updates reps
            llm_decision_transform,      # *Only participating reps will be prompted*
            final_voting_transform,
            resource_allocation
        )

    elif config.mechanism == 'PLD':
        # PLD requires a more complex, multi-phase sequence
        # We model this by chaining multiple decision/update steps.
        # This directly mirrors the (a) -> (b) -> (c) -> (d) -> (e) flow.
        pld_delegation_phase = create_llm_decision_transform(
            agents, llm_service, prompt_logic_type="pld_delegation"
        )
        pld_final_vote_phase = create_llm_decision_transform(
            agents, llm_service, prompt_logic_type="pld_final_vote"
        )
        
        return sequential(
            housekeeping_and_signals,             # (a) Prediction Market (signals)
            pld_delegation_phase,                 # (b) Delegation Voting
            create_delegation_transform(),        # Updates graph based on delegation
            create_power_flow_transform(),        # (c) Vote Update (power calculation)
            pld_final_vote_phase,                 # (d) Resource Voting by power-holders
            final_voting_transform,               # Final aggregation
            resource_allocation                   # (e) Resource Allocation
        )
    else:
        raise ValueError(f"Unknown mechanism: {config.mechanism}")