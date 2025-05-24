import os
import time
import traceback
import pandas as pd
import jax
import jax.random as jr
import jax.numpy as jnp # Ensure jnp is available if _execute_single_simulation_logic uses it directly

from typing import Dict, Any, Tuple, Optional, Callable # Added Callable

# Import your config object and factory functions
from environments.democracy.configuration import (
    PortfolioDemocracyConfig,
    create_thesis_baseline_config,
    create_thesis_highvariance_config # Make sure this is defined in configuration.py
)
# Import necessary functions for simulation logic
from environments.democracy.initialization import initialize_portfolio_democracy_graph_state
from environments.democracy.mechanism_factory import create_portfolio_mechanism_pipeline
from services.llm import ProcessIsolatedLLMService
from core.graph import GraphState # If used in _execute_single_simulation_logic

# --- Mapping from factory name string to actual function ---
CONFIG_FACTORIES: Dict[str, Callable[..., PortfolioDemocracyConfig]] = {
    "create_thesis_baseline_config": create_thesis_baseline_config,
    "create_thesis_highvariance_config": create_thesis_highvariance_config,
    # Add more mappings if you create more factory functions
}

def _execute_single_simulation_logic(
    key: jr.PRNGKey,
    sim_config: PortfolioDemocracyConfig, # Receives the fully prepared config
    llm_service: Optional[ProcessIsolatedLLMService],
    worker_pid: int,
    run_id: int # For more specific logging
) -> pd.DataFrame:
    """
    Core simulation execution logic.
    (This is the corrected version from our previous discussion)
    """
    initial_state = initialize_portfolio_democracy_graph_state(key, sim_config)
    
    llm_instance_for_pipeline = llm_service._service if llm_service and hasattr(llm_service, '_service') else llm_service
    round_transform = create_portfolio_mechanism_pipeline(
        mechanism=sim_config.mechanism,
        llm_service=llm_instance_for_pipeline,
        sim_config=sim_config
    )
    
    simulation_data_list = []
    current_state = initial_state
    
    print(f"[PID {worker_pid}, RunID {run_id}] Starting Core Logic: Mech={sim_config.mechanism}, "
          f"AdvTotalProp={sim_config.agent_settings.adversarial_proportion_total:.2f}, "
          f"Seed={sim_config.seed}, NumRounds={sim_config.num_rounds}")
    
    for round_idx_loop in range(sim_config.num_rounds):
        round_start_time = time.time()
        resources_before = float(current_state.global_attrs.get("current_total_resources", 0.0))
        
        current_state_for_transform = current_state

        try:
            next_state = round_transform(current_state_for_transform)
            transform_success = True
        except Exception as e:
            print(f"[PID {worker_pid}, RunID {run_id}] Error in round {current_state.global_attrs.get('round_num', round_idx_loop)} "
                  f"for config seed {sim_config.seed}: {e}\n{traceback.format_exc()}")
            next_state = current_state_for_transform
            transform_success = False
        
        execution_time = time.time() - round_start_time
        current_state = next_state

        actual_round_completed = int(current_state.global_attrs.get("round_num", round_idx_loop))
        resources_after = float(current_state.global_attrs.get("current_total_resources", 0.0))
        resource_change = resources_after - resources_before
        resource_change_pct = ((resources_after / resources_before) - 1) * 100 if resources_before > 1e-6 else 0.0 # Avoid div by zero
        decision_idx = int(current_state.global_attrs.get("current_decision", -1))
        
        adversarial_influence = 0.0
        if "is_adversarial" in current_state.node_attrs and "voting_power" in current_state.node_attrs:
            adversarial = jnp.asarray(current_state.node_attrs["is_adversarial"])
            voting_power = jnp.asarray(current_state.node_attrs["voting_power"])
            total_power = jnp.sum(voting_power)
            adversarial_power = jnp.sum(voting_power * adversarial)
            if total_power > 1e-6: # Avoid div by zero
                adversarial_influence = float(adversarial_power / total_power)

        round_data_for_df = {
            "round": actual_round_completed,
            "execution_time": execution_time,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "resource_change": resource_change,
            "resource_change_pct": resource_change_pct,
            "adversarial_influence": adversarial_influence,
            "decision_idx": decision_idx,
            "chosen_portfolio": "N/A",
            "transform_success": transform_success,
            "process_id": worker_pid
        }
        if -1 < decision_idx < len(sim_config.portfolios):
            round_data_for_df["chosen_portfolio"] = sim_config.portfolios[decision_idx].name
        
        simulation_data_list.append(round_data_for_df)
        
        # Less frequent logging for parallel runs
        if (actual_round_completed + 1) % (sim_config.num_rounds // 5 if sim_config.num_rounds >=10 else 1) == 0 or \
           actual_round_completed == sim_config.num_rounds - 1 or \
           sim_config.num_rounds <= 5:
             print(f"  [PID {worker_pid}, RunID {run_id}] S{sim_config.seed} R {actual_round_completed+1}/{sim_config.num_rounds} | "
                  f"Res: {resources_after:.2f} | Dec: {round_data_for_df['chosen_portfolio']} | Succ: {transform_success}")

        if resources_after < sim_config.resources.threshold or not transform_success:
            if not transform_success:
                 print(f"  [PID {worker_pid}, RunID {run_id}] S{sim_config.seed} Terminating: transform fail R{actual_round_completed}.")
            else:
                print(f"  [PID {worker_pid}, RunID {run_id}] S{sim_config.seed} Terminating: Res {resources_after:.2f} < thr {sim_config.resources.threshold} R{actual_round_completed+1}.")
            break 
            
    return pd.DataFrame(simulation_data_list)


# This is the function executed by each process in the pool
def run_simulation_task(run_params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]: # Changed from SingleRunParameters for now
    """
    Worker function to run a single simulation configuration.
    `run_params` is expected to be a dictionary representation of `SingleRunParameters`.
    """
    worker_pid = os.getpid()
    worker_start_time = time.time()
    run_id = run_params['run_id'] # For logging

    print(f"[PID {worker_pid}, RunID {run_id}] Task received. Factory: {run_params['config_factory_name']}")

    try:
        # --- Get the appropriate PortfolioDemocracyConfig ---
        factory_name = run_params['config_factory_name']
        config_factory = CONFIG_FACTORIES.get(factory_name)

        if not config_factory:
            raise ValueError(f"Unknown config_factory_name: {factory_name}")

        # Prepare arguments for the chosen config factory
        # The factory functions now primarily take the swept/core params
        factory_args = {
            'mechanism': run_params['mechanism'],
            'adversarial_proportion_total': run_params['adversarial_proportion_total'],
            'seed': run_params['unique_config_seed'],
            # Add other args IF your chosen factory function specifically needs them
            # and they are not already defaults within the factory itself.
            # For create_thesis_baseline_config and create_thesis_highvariance_config (as refactored),
            # other values like num_crops, variance_multiplier etc. are part of *their definition* or
            # passed if create_highvariance calls create_baseline with overrides.
            # The key is that `create_thesis_highvariance_config` would handle its specific num_crops etc.
        }
        # If your factories are structured such that `create_thesis_highvariance_config`
        # sets its specific num_crops, variance_mult etc., and then calls
        # `create_thesis_baseline_config` with THOSE values, then you only need to pass
        # the core swept params to the top-level factory chosen.

        sim_config = config_factory(**factory_args)
        
        # --- JAX key for this specific run ---
        key = jr.PRNGKey(run_params['unique_config_seed']) 
        
        # --- LLM Service Initialization ---
        llm_service: Optional[ProcessIsolatedLLMService] = None
        if run_params.get('llm_model'): # Check if LLM model is specified
            try:
                openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                if openrouter_api_key:
                    llm_service = ProcessIsolatedLLMService(
                        model=run_params['llm_model'],
                        api_key=openrouter_api_key,
                        process_id=f"{worker_pid}-{run_id}"
                    )
                    print(f"[PID {worker_pid}, RunID {run_id}] LLM service for {run_params['llm_model']} initialized.")
                else:
                    print(f"[PID {worker_pid}, RunID {run_id}] OPENROUTER_API_KEY not set. LLM features disabled for this run.")
            except Exception as e_llm:
                print(f"[PID {worker_pid}, RunID {run_id}] Worker LLM init failed: {e_llm}")
                llm_service = None
        
        # --- Execute Simulation ---
        results_df = _execute_single_simulation_logic(key, sim_config, llm_service, worker_pid, run_id)
        
        # --- Attach identifying parameters to each row of the DataFrame ---
        # These come from the input run_params
        df_metadata_cols = {
            'run_id': run_params['run_id'],
            'experiment_name': run_params['experiment_name'],
            'mechanism': run_params['mechanism'],
            'adversarial_proportion_total': run_params['adversarial_proportion_total'],
            'replication_run_index': run_params['replication_run_index'],
            'unique_config_seed': run_params['unique_config_seed']
        }
        for col_name, col_val in df_metadata_cols.items():
            if col_name not in results_df.columns: # Avoid overwriting if already present
                results_df[col_name] = col_val

        # --- Prepare Full Metadata for This Run ---
        final_resources = float(results_df['resources_after'].iloc[-1]) if not results_df.empty else 0.0
        # The `run_params` dictionary itself contains all input parameters for this run.
        full_metadata = {
            **run_params, # Includes run_id, experiment_name, all input params etc.
            'status': 'success',
            'worker_pid': worker_pid,
            'final_resources': final_resources,
            'rounds_completed': len(results_df) if not results_df.empty else 0,
            'simulation_duration_sec': time.time() - worker_start_time,
            'llm_actually_used': llm_service is not None # If LLM was successfully initialized
        }
        print(f"[PID {worker_pid}, RunID {run_id}] Success. Dur: {full_metadata['simulation_duration_sec']:.2f}s. FinRes: {final_resources:.2f}")
        return results_df, full_metadata

    except Exception as e_task:
        error_tb = traceback.format_exc()
        print(f"[PID {worker_pid}, RunID {run_id}] FAILED in run_simulation_task: {e_task}\n{error_tb}")
        # Return all original run_params as part of metadata for complete tracking
        full_metadata = {
            **run_params,
            'status': 'error',
            'worker_pid': worker_pid,
            'error_message': str(e_task),
            'error_traceback': error_tb,
            'simulation_duration_sec': time.time() - worker_start_time,
            'llm_actually_used': False # Assume LLM was not successfully used if task failed here
        }
        # For error DataFrame, include key identifiers
        error_df_content = {k: run_params.get(k) for k in ['run_id', 'experiment_name', 'mechanism', 'adversarial_proportion_total']}
        error_df_content['error'] = str(e_task)
        return pd.DataFrame([error_df_content]), full_metadata