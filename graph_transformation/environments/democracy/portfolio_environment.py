# environments/democracy/portfolio_environment.py
from typing import Dict, List, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import matplotlib.pyplot as plt

from core.graph import GraphState
from core.category import Transform
from execution.call import execute, execute_with_instrumentation
from environments.democracy.portfolio_config import PortfolioDemocracyConfig
from environments.democracy.porfolio_initialization import initialize_portfolio_state
from environments.democracy.portfolio_mechanisms import create_portfolio_mechanism_pipeline

class PortfolioDemocracyEnvironment:
    """Environment for simulating portfolio-based democratic decisions"""
    
    def __init__(self, config: PortfolioDemocracyConfig):
        """Initialize the environment with configuration"""
        self.config = config
        self.key = jr.PRNGKey(config.seed)
        
        # Create mechanism pipelines
        self.pipelines = create_portfolio_mechanism_pipeline()
        self.active_pipeline = self.pipelines[config.mechanism]
        
        # Initialize tracking variables
        self.state_history = []
        self.metrics = {}
    
    def initialize(self) -> GraphState:
        """Initialize the simulation state"""
        self.key, subkey = jr.split(self.key)
        return initialize_portfolio_state(self.config, subkey)
    
    def run_simulation(self) -> Tuple[GraphState, List[GraphState]]:
        """Run a complete simulation"""
        # Initialize state
        initial_state = self.initialize()
        current_state = initial_state
        state_history = [initial_state]
        
        # Execute rounds
        for round_num in range(1, self.config.num_rounds + 1):
            # Split key for this round
            self.key, round_key = jr.split(self.key)
            
            # Update round counter
            current_state = current_state.update_global_attr("round", round_num)
            
            # Execute transformation with instrumentation
            execution_spec = {
                "strategy": "sequential",
                "verify_properties": True,
                "track_history": True,
                "collect_metrics": True,
            }
            
            # Apply mechanism pipeline
            current_state, instrumentation = execute_with_instrumentation(
                self.active_pipeline,
                current_state,
                execution_spec
            )
            
            # Store metrics
            round_metrics = instrumentation["metrics"]
            self.metrics[f"round_{round_num}"] = round_metrics
            
            # Add to history
            state_history.append(current_state)
            
            # Check termination condition
            if current_state.global_attrs.get("total_resources", 0) < self.config.resources.threshold:
                break
        
        # Store final history
        self.state_history = state_history
        
        return current_state, state_history
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Summarize simulation results"""
        if not self.state_history:
            return {"status": "No simulation has been run"}
        
        # Extract key metrics
        final_state = self.state_history[-1]
        initial_state = self.state_history[0]
        
        # Resource trajectory
        resources = [state.global_attrs.get("total_resources", 0) for state in self.state_history]
        
        # Decisions made each round
        decisions = []
        for state in self.state_history[1:]:  # Skip initial state
            decision_idx = state.global_attrs.get("current_decision", -1)
            portfolio_names = list(state.global_attrs.get("portfolios", {}).keys())
            if 0 <= decision_idx < len(portfolio_names):
                decisions.append(portfolio_names[decision_idx])
            else:
                decisions.append("Unknown")
        
        # Calculate summary stats
        summary = {
            "mechanism": self.config.mechanism,
            "num_rounds": len(self.state_history) - 1,
            "initial_resources": initial_state.global_attrs.get("total_resources", 0),
            "final_resources": final_state.global_attrs.get("total_resources", 0),
            "resource_growth": final_state.global_attrs.get("total_resources", 0) / 
                              initial_state.global_attrs.get("total_resources", 1),
            "decisions": decisions,
            "resources": resources,
        }
        
        return summary
    
    def plot_resource_trajectory(self, figsize=(10, 6)):
        """Plot resource trajectory over simulation rounds"""
        if not self.state_history:
            print("No simulation data available")
            return None
        
        resources = [state.global_attrs.get("total_resources", 0) for state in self.state_history]
        rounds = list(range(len(resources)))
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rounds, resources, marker='o', linestyle='-', label=self.config.mechanism)
        
        # Add threshold line
        threshold = self.config.resources.threshold
        ax.axhline(y=threshold, color='r', linestyle='--', 
                  label=f'Survival Threshold ({threshold})')
        
        ax.set_title(f'Resource Trajectory: {self.config.mechanism}')
        ax.set_xlabel('Round')
        ax.set_ylabel('Resources')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig