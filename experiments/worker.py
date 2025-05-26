# experiments/worker.py - SIMPLIFIED FOR TRAJECTORY FOCUS
import os
import time
import traceback
import pandas as pd
import jax
from dataclasses import replace # Added import
import jax.random as jr
import jax.numpy as jnp

from typing import Dict, Any, Tuple, Optional, Callable

# Import your config object and factory functions
from environments.democracy.configuration import (
    PortfolioDemocracyConfig,
    create_thesis_baseline_config,
    create_thesis_highvariance_config
)
from environments.democracy.initialization import initialize_portfolio_democracy_graph_state
from environments.democracy.mechanism_factory import create_portfolio_mechanism_pipeline
from services.llm import ProcessIsolatedLLMService
from core.graph import GraphState

# --- Mapping from factory name string to actual function ---
CONFIG_FACTORIES: Dict[str, Callable[..., PortfolioDemocracyConfig]] = {
    "create_thesis_baseline_config": create_thesis_baseline_config,
    "create_thesis_highvariance_config": create_thesis_highvariance_config,
}

def _execute_trajectory_simulation(
    key: jr.PRNGKey,
    sim_config: PortfolioDemocracyConfig,
    llm_service: Optional[ProcessIsolatedLLMService],
    worker_pid: int,
    run_id: int
) -> pd.DataFrame:
    """
    CORE SIMULATION: Simplified execution focused on resource trajectory.
    
    TRAJECTORY TRACKING:
    - Round number
    - Resource level after each round
    - Basic run identification
    
    ELIMINATES:
    - Execution time tracking
    - Adversarial influence calculation
    - Decision quality metrics
    - Portfolio selection details
    - Transform success monitoring
    - Complex debugging output
    
    DEBUGGING STRATEGY:
    - Minimal round-level progress (every 10th round)
    - Resource progression summary
    - Simple termination logging
    """
    # Initialize simulation
    initial_state = initialize_portfolio_democracy_graph_state(key, sim_config)
    
    llm_instance = llm_service._service if llm_service and hasattr(llm_service, '_service') else llm_service
    round_transform = create_portfolio_mechanism_pipeline(
        mechanism=sim_config.mechanism,
        llm_service=llm_instance,
        sim_config=sim_config
    )
    
    # Trajectory storage
    trajectory_points = []
    current_state = initial_state
    
    print(f"[PID {worker_pid}] Run {run_id}: {sim_config.mechanism} simulation started")
    
    # Execute simulation rounds
    for round_idx in range(sim_config.num_rounds):
        # Apply round transformation
        try:
            next_state = round_transform(current_state)
            current_state = next_state
            simulation_success = True
        except Exception as e:
            print(f"[PID {worker_pid}] Run {run_id}: Round {round_idx} failed: {e}")
            simulation_success = False
            break
        
        # Extract trajectory data
        actual_round = int(current_state.global_attrs.get("round_num", round_idx))
        resources_after = float(current_state.global_attrs.get("current_total_resources", 0.0))
        chosen_portfolio_idx = int(current_state.global_attrs.get("current_decision", -1)) # Get chosen portfolio
        
        # Store trajectory point
        trajectory_points.append({
            'round': actual_round,
            'resources_after': resources_after,
            'chosen_portfolio_idx': chosen_portfolio_idx # Add chosen portfolio index
        })
        
        # Simple progress reporting (every 10th round)
        if (actual_round + 1) % 10 == 0 or actual_round == sim_config.num_rounds - 1:
            print(f"[PID {worker_pid}] Run {run_id}: Round {actual_round+1}/{sim_config.num_rounds}, "
                  f"Resources: {resources_after:.1f}")
        
        # Check termination condition
        if resources_after < sim_config.resources.threshold:
            print(f"[PID {worker_pid}] Run {run_id}: Terminated at round {actual_round+1} "
                  f"(resources {resources_after:.1f} < threshold {sim_config.resources.threshold})")
            break
    
    # Create trajectory DataFrame
    if trajectory_points:
        trajectory_df = pd.DataFrame(trajectory_points)
        print(f"[PID {worker_pid}] Run {run_id}: Completed with {len(trajectory_df)} trajectory points")
    else:
        # Empty trajectory for failed simulations
        trajectory_df = pd.DataFrame(columns=['round', 'resources_after', 'chosen_portfolio_idx'])
        print(f"[PID {worker_pid}] Run {run_id}: No trajectory data generated")
    
    return trajectory_df


def run_simulation_task(run_params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    SIMPLIFIED SIMULATION TASK: Trajectory-focused simulation execution.
    
    ARCHITECTURAL SIMPLIFICATION:
    - Returns only essential trajectory data (round, resources_after)
    - Minimal metadata for run identification
    - Simple error handling with basic logging
    - No complex performance metrics or debugging data
    
    RETURN STRUCTURE:
    - DataFrame: trajectory data with run identification columns
    - Dict: basic metadata for results aggregation
    
    ELIMINATES:
    - Complex timeline metadata attachment
    - Comprehensive error reporting structures
    - Performance tracking and optimization
    - Detailed debugging information
    """
    worker_pid = os.getpid()
    worker_start_time = time.time()
    run_id = run_params.get('run_id', -1)
    
    print(f"[PID {worker_pid}] Starting Run {run_id}")
    
    try:
        # CONFIGURATION SETUP
        factory_name = run_params.get('config_factory_name', 'create_thesis_baseline_config')
        config_factory = CONFIG_FACTORIES.get(factory_name)
        
        if not config_factory:
            raise ValueError(f"Unknown config factory: {factory_name}")
        
        # Create simulation configuration using the factory.
        # The factory now accepts 'adversarial_proportion_total'.
        sim_config: PortfolioDemocracyConfig = config_factory(
            mechanism=run_params['mechanism'],
            adversarial_proportion_total=run_params['adversarial_proportion_total'],
            seed=run_params['unique_config_seed']
        )        

        
        # JAX random key
        key = jr.PRNGKey(run_params['unique_config_seed'])
        
        # LLM SERVICE (optional)
        llm_service = None
        if run_params.get('llm_model'):
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                try:
                    llm_service = ProcessIsolatedLLMService(
                        model=run_params['llm_model'],
                        api_key=api_key,
                        process_id=f"{worker_pid}-{run_id}"
                    )
                except Exception as e:
                    print(f"[PID {worker_pid}] LLM init failed: {e}")
        
        # EXECUTE SIMULATION
        trajectory_df = _execute_trajectory_simulation(key, sim_config, llm_service, worker_pid, run_id)
        
        # ADD RUN IDENTIFICATION to trajectory data
        if not trajectory_df.empty:
            trajectory_df['run_id'] = run_params['run_id']
            trajectory_df['mechanism'] = run_params['mechanism']
            trajectory_df['adversarial_proportion_total'] = run_params['adversarial_proportion_total']
            trajectory_df['replication_run_index'] = run_params['replication_run_index']
            trajectory_df['experiment_name'] = run_params.get('experiment_name', 'unknown')
        
        # BASIC METADATA
        final_resources = float(trajectory_df['resources_after'].iloc[-1]) if not trajectory_df.empty else 0.0
        simulation_duration = time.time() - worker_start_time
        
        metadata = {
            'run_id': run_params['run_id'],
            'mechanism': run_params['mechanism'],
            'adversarial_proportion_total': run_params['adversarial_proportion_total'],
            'replication_run_index': run_params['replication_run_index'],
            'status': 'success',
            'worker_pid': worker_pid,
            'final_resources': final_resources,
            'rounds_completed': len(trajectory_df),
            'simulation_duration_sec': simulation_duration
        }
        
        print(f"[PID {worker_pid}] Run {run_id}: SUCCESS in {simulation_duration:.1f}s, "
              f"Final resources: {final_resources:.2f}")
        
        return trajectory_df, metadata
    
    except Exception as e:
        # SIMPLE ERROR HANDLING
        error_duration = time.time() - worker_start_time
        error_message = str(e)
        
        print(f"[PID {worker_pid}] Run {run_id}: FAILED after {error_duration:.1f}s: {error_message}")
        
        # Create minimal error data
        error_trajectory = pd.DataFrame([{
            'round': 0,
            'resources_after': 0.0,
            'run_id': run_params.get('run_id', -1),
            'mechanism': run_params.get('mechanism', 'unknown'), # Ensure mechanism is included
            'chosen_portfolio_idx': -1, # Default for error cases
            'adversarial_proportion_total': run_params.get('adversarial_proportion_total', 0.0),
            'replication_run_index': run_params.get('replication_run_index', 0),
            'experiment_name': run_params.get('experiment_name', 'unknown')
        }])
        
        error_metadata = {
            'run_id': run_params.get('run_id', -1),
            'mechanism': run_params.get('mechanism', 'unknown'),
            'adversarial_proportion_total': run_params.get('adversarial_proportion_total', 0.0),
            'replication_run_index': run_params.get('replication_run_index', 0),
            'status': 'error',
            'worker_pid': worker_pid,
            'error_message': error_message,
            'final_resources': 0.0,
            'rounds_completed': 0,
            'simulation_duration_sec': error_duration
        }
        
        return error_trajectory, error_metadata


# CONVENIENCE FUNCTION for testing single simulations
def test_single_simulation(
    mechanism: str = "PLD",
    adversarial_proportion: float = 0.3,
    seed: int = 42,
    config_factory: str = "create_thesis_baseline_config"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    TEST FUNCTION: Run a single simulation for debugging/validation.
    
    USAGE:
    ```python
    trajectory_df, metadata = test_single_simulation(
        mechanism="PLD",
        adversarial_proportion=0.3,
        seed=42
    )
    print(f"Generated {len(trajectory_df)} trajectory points")
    ```
    """
    test_params = {
        'run_id': 0,
        'mechanism': mechanism,
        'adversarial_proportion_total': adversarial_proportion,
        'replication_run_index': 0,
        'unique_config_seed': seed,
        'config_factory_name': config_factory,
        'experiment_name': 'test_run',
        'llm_model': None  # No LLM for testing
    }
    
    return run_simulation_task(test_params)


if __name__ == "__main__":
    # Quick test of trajectory simulation
    print("Testing trajectory simulation...")
    
    trajectory_data, metadata = test_single_simulation()
    
    if not trajectory_data.empty:
        print(f"SUCCESS: Generated {len(trajectory_data)} trajectory points")
        print(f"Final resources: {metadata['final_resources']:.2f}")
        print(f"Trajectory preview:")
        print(trajectory_data.head())
    else:
        print("FAILED: No trajectory data generated")
        
    print("Test complete.")