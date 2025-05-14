# For: environments/democracy/portfolio_environment.py
# Based on input_file_11.py (stub) and requirements for a simulation environment class

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, List, Any, Optional, Tuple

from core.graph import GraphState
from core.category import Transform
from execution.simulation import run_simulation # Assumes this function is available

from environments.democracy.configuration import PortfolioDemocracyConfig
from environments.democracy.portfolio_initialization import initialize_portfolio_state
from environments.democracy.portfolio_mechanism_factory import create_portfolio_mechanism_pipeline
from services.adapters import AnalysisAdapter, RuleBasedAnalysisAdapter, LLMAnalysisAdapter # For type hinting and default
from services.llm import LLMService, OpenAIService # For default LLM adapter

class PortfolioDemocracyEnvironment:
    """
    Environment for running portfolio democracy simulations.
    This class manages configuration, state initialization, pipeline creation,
    and simulation execution.
    """
    
    def __init__(self, config: PortfolioDemocracyConfig, analysis_adapter: Optional[AnalysisAdapter] = None):
        """
        Initialize the portfolio democracy environment.

        Args:
            config: The configuration object for the simulation.
            analysis_adapter: The portfolio analysis adapter to use. 
                              If None, defaults to RuleBasedAnalysisAdapter.
        """
        self.config = config
        
        if analysis_adapter is None:
            # Defaulting to RuleBased. If LLM is desired as default, it needs an LLMService.
            # print("No analysis adapter provided, defaulting to RuleBasedAnalysisAdapter.")
            self.analysis_adapter = RuleBasedAnalysisAdapter()
        else:
            self.analysis_adapter = analysis_adapter
            
        self.pipeline: Optional[Transform] = None
        self.state_history: List[GraphState] = []
        self.metrics_history: List[Dict[str, Any]] = []


    def _ensure_pipeline(self):
        """Ensures the transformation pipeline is created."""
        if self.pipeline is None:
            self.pipeline = create_portfolio_mechanism_pipeline(
                mechanism=self.config.mechanism,
                config=self.config,
                analysis_adapter=self.analysis_adapter
            )
            if self.config.jit_compile:
                # print("JIT compiling the pipeline...")
                self.pipeline = jax.jit(self.pipeline)


    def initialize_simulation_state(self, key: jr.PRNGKey) -> GraphState:
        """
        Initializes the graph state for the simulation.

        Args:
            key: JAX PRNG key for initialization.

        Returns:
            The initial GraphState.
        """
        return initialize_portfolio_state(self.config, key)

    def run_single_trial(
        self, 
        key: jr.PRNGKey, 
        execution_spec: Optional[Dict[str, Any]] = None
    ) -> Tuple[GraphState, List[GraphState]]:
        """
        Run a single simulation trial.

        Args:
            key: JAX PRNG key for this trial.
            execution_spec: Optional dictionary specifying execution parameters for run_simulation.

        Returns:
            A tuple containing the final GraphState and the history of states.
        """
        self._ensure_pipeline()
        key_init, key_sim = jr.split(key)
        
        initial_state = self.initialize_simulation_state(key_init)
        
        # Define termination condition
        def default_termination_condition(state: GraphState) -> bool:
            resources = state.global_attrs.get("total_resources", self.config.resources.initial_amount)
            threshold = state.global_attrs.get("resource_threshold", self.config.resources.threshold)
            # Check if resources is a JAX array or Python float
            if isinstance(resources, jnp.ndarray): resources = resources.item()
            if isinstance(threshold, jnp.ndarray): threshold = threshold.item()
            return resources < threshold

        effective_execution_spec = {
            "strategy": "sequential", # Default, can be overridden by execution_spec
            "verify_properties": False, # Typically False for speed in multiple runs
            "track_history": True, # Needed for analysis
            "collect_metrics": True, # If execution.call supports this directly
            **(execution_spec or {})
        }
        
        final_state, state_history = run_simulation(
            initial_state=initial_state,
            transform=self.pipeline, # JIT-compiled if configured
            num_rounds=self.config.num_rounds,
            key=key_sim,
            execution_spec=effective_execution_spec, # Pass spec to run_simulation
            termination_condition=default_termination_condition
        )
        
        self.state_history = state_history # Store history of the last run trial
        self._collect_trial_metrics(final_state, state_history) # Collect metrics from this trial

        return final_state, state_history

    def _collect_trial_metrics(self, final_state: GraphState, state_history: List[GraphState]):
        """Basic metric collection for a trial."""
        # Example metrics: more can be added
        final_resources = final_state.global_attrs.get("total_resources", 0.0)
        if isinstance(final_resources, jnp.ndarray): final_resources = final_resources.item()

        initial_resources = self.config.resources.initial_amount
        
        metrics = {
            "final_resources": final_resources,
            "resource_growth_factor": final_resources / initial_resources if initial_resources > 0 else 0,
            "rounds_completed": final_state.global_attrs.get("round", 0),
            "survived": final_resources >= self.config.resources.threshold,
            "decisions_history": [s.global_attrs.get("current_decision") for s in state_history[1:]], # Skip initial state
            "resource_trajectory": [s.global_attrs.get("total_resources",0.0).item() if isinstance(s.global_attrs.get("total_resources"), jnp.ndarray) else s.global_attrs.get("total_resources",0.0) for s in state_history]
        }
        self.metrics_history.append(metrics)


    def run_multiple_trials(
        self, 
        num_trials: int, 
        base_key: jr.PRNGKey,
        execution_spec: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple simulation trials and collect metrics.

        Args:
            num_trials: The number of trials to run.
            base_key: Base JAX PRNG key for generating trial-specific keys.
            execution_spec: Execution specification for each trial.

        Returns:
            A list of metrics dictionaries, one for each trial.
        """
        self.metrics_history = [] # Reset for new set of trials
        trial_keys = jr.split(base_key, num_trials)
        
        for i in range(num_trials):
            print(f"Running trial {i+1}/{num_trials}...")
            self.run_single_trial(trial_keys[i], execution_spec)
            # Metrics are collected by _collect_trial_metrics and appended to self.metrics_history
            
        return self.get_aggregated_metrics()

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Aggregates metrics from all run trials."""
        if not self.metrics_history:
            return {"message": "No trials run or no metrics collected."}

        num_trials_completed = len(self.metrics_history)
        avg_final_resources = sum(m["final_resources"] for m in self.metrics_history) / num_trials_completed
        avg_growth_factor = sum(m["resource_growth_factor"] for m in self.metrics_history) / num_trials_completed
        survival_rate = sum(1 for m in self.metrics_history if m["survived"]) / num_trials_completed
        avg_rounds = sum(m["rounds_completed"] for m in self.metrics_history) / num_trials_completed

        return {
            "num_trials": num_trials_completed,
            "avg_final_resources": avg_final_resources,
            "avg_resource_growth_factor": avg_growth_factor,
            "survival_rate": survival_rate,
            "avg_rounds_completed": avg_rounds,
            "all_trial_metrics": self.metrics_history # Optionally include all raw trial data
        }


if __name__ == '__main__':
    # Example Usage:
    # 1. Create a configuration
    sim_config = PortfolioDemocracyConfig(
        num_agents=10, 
        num_rounds=5, 
        seed=42,
        mechanism="PLD",
        jit_compile=False # JIT can be slower for very small runs / first compilation
    )

    # 2. Create an Analysis Adapter (e.g., RuleBased or LLM with a Mock/Real Service)
    # rule_adapter = RuleBasedAnalysisAdapter()
    
    # For LLM (assuming OPENAI_API_KEY is set or MockLLMService is available)
    try:
        from services.llm import OpenAIService # Or your MockLLMService for testing without API calls
        # For real LLM (ensure API key is available)
        # llm_service = OpenAIService() 
        # For mock:
        class MockLLMService(LLMService): # Redefine here for standalone example if needed
            def generate(self, prompt: str) -> str:
                return '{"Conservative": 7.0, "Balanced": 8.0, "Aggressive": 6.0, "Contrarian": 3.0, "Market-Weighted": 7.0}'
        llm_service = MockLLMService(api_key="dummy_key_not_used_by_mock")
        llm_adapter = LLMAnalysisAdapter(llm_service=llm_service)
        selected_adapter = llm_adapter
        print("Using LLMAdapter (Mock) for the test run.")
    except ImportError as e:
        print(f"LLM components not available ({e}), defaulting to RuleBasedAnalysisAdapter.")
        selected_adapter = RuleBasedAnalysisAdapter()
    except Exception as e: # Catch other LLM service init errors
        print(f"Error initializing LLM service ({e}), defaulting to RuleBasedAnalysisAdapter.")
        selected_adapter = RuleBasedAnalysisAdapter()


    # 3. Create the environment
    env = PortfolioDemocracyEnvironment(config=sim_config, analysis_adapter=selected_adapter)

    # 4. Run multiple trials
    master_key = jr.PRNGKey(sim_config.seed)
    print(f"\nStarting simulation for mechanism: {sim_config.mechanism} with {selected_adapter.__class__.__name__}")
    aggregated_results = env.run_multiple_trials(num_trials=2, base_key=master_key) # Run 2 trials

    # 5. Print aggregated results
    print("\n--- Aggregated Results ---")
    for key, value in aggregated_results.items():
        if key != "all_trial_metrics":
            print(f"{key}: {value}")
    
    # Optionally print details of the first trial's resource trajectory
    if aggregated_results.get("all_trial_metrics"):
        first_trial_trajectory = aggregated_results["all_trial_metrics"][0].get("resource_trajectory")
        if first_trial_trajectory:
            print(f"\nResource Trajectory (Trial 1): {first_trial_trajectory}")

print("environments/democracy/portfolio_environment.py content generated.")