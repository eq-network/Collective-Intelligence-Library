# environments/democracy/simulation.py
"""
The core simulation engine for the Portfolio Democracy environment.
This module defines the Environment class that manages state, orchestrates
agent interactions, and applies world dynamics.
"""
from typing import List, Dict, Any, Optional, Tuple
import jax.numpy as jnp
import jax.random as jr

from core.environment import Environment
from core.agents import Agent
from core.graph import GraphState
from core.category import sequential, Transform
from services.llm import LLMService

# Import your configurations and transformations
from engine.environments.democracy.random.configuration import PortfolioDemocracyConfig # Using this as the main config type
from engine.environments.democracy.random.initialization import initialize_agent_population, initialize_graph_state
from engine.environments.democracy.random.mechanism_factory import (
    create_start_of_round_housekeeping_transform,
    create_prediction_market_transform,
    create_delegation_transform,
    create_power_flow_transform,
    create_voting_transform,
    create_resource_transform
)

class PortfolioDemocracyEnvironment(Environment):
    """
    Manages the state and dynamics of the portfolio democracy simulation.
    This class orchestrates the simulation loop, replacing the old factory model.
    """
    def __init__(self, config: PortfolioDemocracyConfig, llm_service: Optional[LLMService] = None):
        self.config = config
        self.llm_service = llm_service
        self.max_rounds = config.num_rounds
        
        # Initialize agents and state
        agents = initialize_agent_population(config, self.llm_service)
        initial_state = initialize_graph_state(config)
        
        super().__init__(agents, initial_state)
        
        # Compose the world update transformation pipeline
        self.world_transform = self._build_world_transform()
        
    def _build_world_transform(self) -> Transform:
        """Assembles the sequence of pure transformations for a simulation round."""
        pipeline_steps = [
            create_start_of_round_housekeeping_transform(),
            create_prediction_market_transform(self._get_agent_specific_signals),
        ]

        if self.config.mechanism == "PLD":
            pipeline_steps.extend([
                create_delegation_transform(),
                create_power_flow_transform(),
            ])
        
        pipeline_steps.extend([
            create_voting_transform(self._aggregate_votes_by_mechanism),
            create_resource_transform(self._calculate_resource_change)
        ])
        
        return sequential(*pipeline_steps)
        
    def get_observation_for_agent(self, agent: Agent) -> Dict[str, Any]:
        """Constructs the observation dict for a single agent from the current state."""
        agent_id = agent.agent_id
        
        # Get agent-specific noisy signals
        all_signals = self.state.global_attrs.get("agent_specific_prediction_signals", {})
        agent_signals = all_signals.get(agent_id, jnp.array([]))
        
        # Format portfolio options string
        portfolio_options_str = "No portfolio options available."
        portfolios = self.state.global_attrs.get("portfolios", [])
        if portfolios and agent_signals.size > 0:
            portfolio_lines = []
            for p_cfg in portfolios:
                p_weights = jnp.array(p_cfg.weights)
                expected_yield = jnp.sum(p_weights * agent_signals)
                portfolio_lines.append(f"- {p_cfg.name}: {p_cfg.description} (Predicted Yield: {expected_yield:.3f}x)")
            portfolio_options_str = "\n".join(portfolio_lines)
            
        # Format delegate history string for PLD
        delegate_history_str = "N/A"
        if self.config.mechanism == "PLD":
            # (Future enhancement: Add real performance history here)
            delegate_history_str = "Performance tracking is under development."
            
        return {
            "round_num": self.state.global_attrs.get("round_num", 0),
            "portfolio_options_str": portfolio_options_str,
            "delegate_history_str": delegate_history_str,
        }

    def apply_actions(self, actions: List[Action]) -> GraphState:
        """Packs agent actions into the GraphState before applying world transforms."""
        num_agents = len(self.agents)
        num_portfolios = len(self.config.portfolios)
        
        # Initialize arrays for votes and delegations
        agent_votes = jnp.zeros((num_agents, num_portfolios), dtype=jnp.int32)
        delegation_targets = -jnp.ones(num_agents, dtype=jnp.int32)

        for i, action in enumerate(actions):
            if action.get("type") == "vote":
                agent_votes = agent_votes.at[i].set(jnp.array(action.get("votes", [0]*num_portfolios)))
            elif action.get("type") == "delegate":
                delegation_targets = delegation_targets.at[i].set(action.get("target_id", -1))

        # Update the state with the packed actions
        new_node_attrs = self.state.node_attrs.copy()
        new_node_attrs["agent_portfolio_votes"] = agent_votes
        new_node_attrs["delegation_target"] = delegation_targets
        
        state_with_actions = self.state.replace(node_attrs=new_node_attrs)
        
        # Apply the world transformation pipeline
        return self.world_transform(state_with_actions)

    def is_terminated(self) -> bool:
        """Checks for termination conditions."""
        resources = self.state.global_attrs["current_total_resources"]
        if resources < self.config.resources.threshold:
            return True
        if self.round_num >= self.max_rounds:
            return True
        return False

    def reset(self):
        """Resets the environment to its initial state for a new run."""
        initial_state = initialize_graph_state(self.config)
        self.state = initial_state
        self.round_num = 0
        self.history = [self.state]

    # --- Helper methods for transformations ---
    def _get_agent_specific_signals(self, state: GraphState, config: Dict) -> Dict:
        """Generates noisy signals for each agent based on their cognitive resources."""
        # (This logic can be copied from the old mechanism_factory)
        # ... returns dict of {agent_id: signals_array}
        pass # Placeholder for brevity

    def _aggregate_votes_by_mechanism(self, state: GraphState, config: Dict) -> jnp.ndarray:
        """Aggregates votes based on the active democratic mechanism."""
        # (This logic can be copied from the old mechanism_factory)
        # ...
        pass # Placeholder for brevity
    
    def _calculate_resource_change(self, state: GraphState, config: Dict) -> float:
        """Calculates resource change factor based on the collective decision."""
        # (This logic can be copied from the old _portfolio_resource_calculator)
        # ...
        pass # Placeholder for brevity