# environments/democracy/environment.py
from typing import Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState
from core.category import Transform, sequential
from execution.call import execute, execute_with_instrumentation
from environments.democracy.configuration import DemocraticEnvironmentConfig
from environments.democracy.initialization import initialize_democratic_graph_state
from environments.democracy.mechanism_factory import create_mechanism_pipeline

class DemocraticEnvironment:
    """
    Environment for simulating democratic decision mechanisms.
    
    This class handles the configuration, initialization, execution, and
    analysis of democratic mechanism simulations, providing a high-level
    interface that maps directly to research parameters.
    """
    
    def __init__(self, config: DemocraticEnvironmentConfig):
        """Initialize environment with configuration."""
        self.config = config
        self.key = jr.PRNGKey(config.seed)
        self.mechanism_pipeline = create_mechanism_pipeline(config.mechanism, config)
        self.state_history = []
        self.metrics = []
        
    def initialize(self) -> GraphState:
        """Initialize graph state from configuration."""
        # Split key for initialization
        self.key, subkey = jr.split(self.key)
        
        # Map configuration to initialization parameters
        return initialize_democratic_graph_state(
            num_agents=self.config.num_agents,
            crops=self.config.resources.options,
            initial_resources=self.config.resources.initial_amount,
            resource_min_threshold=self.config.resources.threshold,
            adversarial_proportion=self.config.adversarial.proportion,
            adversarial_introduction=self.config.adversarial.introduction,
            yield_volatility=self.config.resources.yield_volatility,
            key=subkey,
            network_params={
                "type": self.config.network.topology,
                "p_connect": self.config.network.avg_connections / self.config.num_agents,
                "k": self.config.network.avg_connections,
                "p_rewire": self.config.network.clustering_coefficient,
            }
        )
    
    def run_simulation(self) -> Tuple[GraphState, List[GraphState]]:
        """Run a single simulation trial."""
        # Initialize state
        initial_state = self.initialize()
        
        # Split key for simulation
        self.key, subkey = jr.split(self.key)
        
        # Create execution specification
        execution_spec = {
            "strategy": "sequential",
            "verify_properties": True,
            "track_history": True,
            "collect_metrics": True,
            "manage_memory": True,
        }
        
        # Apply JIT if configured
        if self.config.jit_compile:
            self.mechanism_pipeline = jax.jit(self.mechanism_pipeline)
        
        # State history for tracking
        state_history = [initial_state]
        current_state = initial_state
        
        # Execute simulation rounds
        for round_num in range(1, self.config.num_rounds + 1):
            # Update round counter
            current_state = current_state.update_global_attr("round", round_num)
            
            # Execute transformation via execution call
            current_state, instrumentation = execute_with_instrumentation(
                self.mechanism_pipeline,
                current_state,
                execution_spec
            )
            
            # Record state
            state_history.append(current_state)
            
            # Check termination condition
            if current_state.global_attrs.get("total_resources", float('inf')) < \
               current_state.global_attrs.get("resource_min_threshold", 0):
                break
        
        # Store history
        self.state_history = state_history
        
        return current_state, state_history
    
    def run_multiple_trials(self, num_trials: Optional[int] = None) -> Dict[str, Any]:
        """Run multiple simulation trials and aggregate results."""
        num_trials = num_trials or self.config.num_trials
        all_metrics = []
        
        for _ in range(num_trials):
            final_state, state_history = self.run_simulation()
            metrics = self.calculate_metrics(final_state, state_history)
            all_metrics.append(metrics)
        
        # Aggregate metrics across trials
        aggregated = self.aggregate_metrics(all_metrics)
        self.metrics = aggregated
        
        return aggregated
    
    def calculate_metrics(self, final_state: GraphState, 
                          state_history: List[GraphState]) -> Dict[str, Any]:
        """Calculate metrics for a single simulation trial."""
        from environments.democracy.simulation_metrics import calculate_simulation_metrics
        return calculate_simulation_metrics(
            final_state=final_state,
            state_history=state_history,
            config=self.config
        )
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple trials."""
        from environments.democracy.simulation_metrics import aggregate_simulation_metrics
        return aggregate_simulation_metrics(metrics_list)