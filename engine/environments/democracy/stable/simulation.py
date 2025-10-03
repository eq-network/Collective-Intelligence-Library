# environments/democracy/stable/simulation.py
"""
The main simulation orchestrator for the Stable Democracy environment.
"""
import importlib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Core interfaces and data structures
from core.agents import Agent, Action
from core.graph import GraphState

# Environment-specific components
from .configuration import StablePortfolioDemocracyConfig
from .initialization import initialize_graph_state
from .mechanism_pipeline import create_stable_democracy_pipeline
from services.llm import LLMService

def _import_class(class_path: str) -> type:
    """Dynamically imports a class from a full Python path string."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class StableDemocracySimulation:
    """
    Orchestrates a complete simulation run for the stable democracy environment.
    It takes a configuration, builds the world and its inhabitants, and runs the simulation.
    """
    def __init__(self, config: StablePortfolioDemocracyConfig):
        print(f"Initializing simulation for mechanism: {config.mechanism} with seed: {config.seed}")
        self.config = config
        self.llm_service = LLMService(model=config.llm_model) if config.llm_model else None

        # 1. Instantiate Agent Population from config paths
        self.agents = self._initialize_agents()

        # 2. Initialize the GraphState
        self.state = initialize_graph_state(config, self.agents)
        # Store a reference to the config in the state for easy access by transforms
        self.state = self.state.replace(global_attrs={**self.state.global_attrs, 'config_ref': config})
        
        # 3. Build the Round Transformation Pipeline
        self.round_pipeline = create_stable_democracy_pipeline(config, self.agents)
        
        self.history: List[Dict[str, Any]] = []
        print("Simulation initialization complete.")

    def _initialize_agents(self) -> List[Agent]:
        """Dynamically creates agent instances from the class paths in the config."""
        agents = []
        num_adv = int(round(self.config.num_agents * self.config.agent_settings.adversarial_proportion_total))
        
        AlignedAgentClass = _import_class(self.config.aligned_agent_class_path)
        AdversarialAgentClass = _import_class(self.config.adversarial_agent_class_path)
        
        roles = ["Delegate"] * self.config.num_delegates + ["Voter"] * (self.config.num_agents - self.config.num_delegates)
        np.random.RandomState(self.config.seed).shuffle(roles)

        for i in range(self.config.num_agents):
            agent_class = AdversarialAgentClass if i < num_adv else AlignedAgentClass
            # The agent constructor now receives all the context it needs
            agents.append(agent_class(
                agent_id=i,
                role=roles[i],
                llm_service=self.llm_service,
                mechanism=self.config.mechanism,
                num_portfolios=len(self.config.portfolios)
            ))
        print(f"Initialized {len(agents)} agents: {num_adv} adversarial, {len(agents)-num_adv} aligned.")
        return agents

    def run(self) -> pd.DataFrame:
        """Executes the full simulation from start to finish and returns the results."""
        self.history = [self.get_state_summary()] # Log initial state

        for i in range(self.config.num_rounds):
            print(f"\n--- Round {i+1}/{self.config.num_rounds} ---")
            
            # The pipeline function does all the work for one round
            self.state = self.round_pipeline(self.state)
            
            # Log results and check for termination
            self.history.append(self.get_state_summary())
            if self.is_terminated():
                print(f"Simulation terminated early at round {i+1}.")
                break
                
        return pd.DataFrame(self.history)

    def get_state_summary(self) -> Dict[str, Any]:
        """Extracts key metrics from the current state for logging."""
        return {
            'round': self.state.global_attrs.get("round_num", 0),
            'resources_after': self.state.global_attrs.get("current_total_resources", 0.0),
            'chosen_portfolio_idx': self.state.global_attrs.get("current_decision", -1),
        }

    def is_terminated(self) -> bool:
        """Checks if the simulation should terminate."""
        resources = self.state.global_attrs.get("current_total_resources")
        if resources < self.config.survival_threshold:
            return True
        return False