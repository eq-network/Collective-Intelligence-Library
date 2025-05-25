import os
import time
import traceback
import pandas as pd
import jax
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

def _execute_single_simulation_logic(
    key: jr.PRNGKey,
    sim_config: PortfolioDemocracyConfig,
    llm_service: Optional[ProcessIsolatedLLMService],
    worker_pid: int,
    run_id: int
) -> pd.DataFrame:
    """
    Core simulation with CLEAN progress tracking.
    
    DEBUG STRATEGY:
    - Clear round-level progress information
    - Resource trajectory tracking
    - Decision outcome summaries
    - Minimal technical noise
    """
    initial_state = initialize_portfolio_democracy_graph_state(key, sim_config)
    
    llm_instance_for_pipeline = llm_service._service if llm_service and hasattr(llm_service, '_service') else llm_service
    round_transform = create_portfolio_mechanism_pipeline(
        mechanism=sim_config.mechanism,
        llm_service=llm_instance_for_pipeline,
        sim_config=sim_config
    )
    
    timeline_data_list = []
    current_state = initial_state
    
    print(f"[PID {worker_pid}, RunID {run_id}] {sim_config.mechanism} simulation started")
    print(f"  Adversarial: {sim_config.agent_settings.adversarial_proportion_total:.1%}, "
          f"Agents: {sim_config.num_agents}, Rounds: {sim_config.num_rounds}")
    
    for round_idx_loop in range(sim_config.num_rounds):
        round_start_time = time.time()
        resources_before = float(current_state.global_attrs.get("current_total_resources", 0.0))
        
        # Execute round transformation
        try:
            next_state = round_transform(current_state)
            transform_success = True
        except Exception as e:
            print(f"[PID {worker_pid}, RunID {run_id}] R{round_idx_loop} ERROR: {e}")
            next_state = current_state
            transform_success = False
        
        execution_time = time.time() - round_start_time
        current_state = next_state

        # Extract round results
        actual_round_completed = int(current_state.global_attrs.get("round_num", round_idx_loop))
        resources_after = float(current_state.global_attrs.get("current_total_resources", 0.0))
        resource_change = resources_after - resources_before
        resource_change_pct = ((resources_after / resources_before) - 1) * 100 if resources_before > 1e-6 else 0.0
        decision_idx = int(current_state.global_attrs.get("current_decision", -1))
        
        # Calculate adversarial influence
        adversarial_influence = 0.0
        if "is_adversarial" in current_state.node_attrs and "voting_power" in current_state.node_attrs:
            adversarial = jnp.asarray(current_state.node_attrs["is_adversarial"])
            voting_power = jnp.asarray(current_state.node_attrs["voting_power"])
            total_power = jnp.sum(voting_power)
            if total_power > 1e-6:
                adversarial_power = jnp.sum(voting_power * adversarial)
                adversarial_influence = float(adversarial_power / total_power)

        # Get portfolio name
        chosen_portfolio_name = "N/A"
        if 0 <= decision_idx < len(sim_config.portfolios):
            chosen_portfolio_name = sim_config.portfolios[decision_idx].name

        # Store timeline data
        round_timeline_data = {
            "round": actual_round_completed,
            "execution_time": execution_time,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "resource_change": resource_change,
            "resource_change_pct": resource_change_pct,
            "decision_idx": decision_idx,
            "chosen_portfolio": chosen_portfolio_name,
            "adversarial_influence": adversarial_influence,
            "transform_success": transform_success,
            "process_id": worker_pid
        }
        
        timeline_data_list.append(round_timeline_data)
        
        # CLEAN PROGRESS: Show meaningful round completion info
        if (actual_round_completed + 1) % max(1, sim_config.num_rounds // 10) == 0 or actual_round_completed == sim_config.num_rounds - 1:
            print(f"  [PID {worker_pid}] R{actual_round_completed+1:2d}/{sim_config.num_rounds} "
                  f"Resources: {resources_after:8.1f} ({resource_change_pct:+6.1f}%) "
                  f"Portfolio: {chosen_portfolio_name[:15]:<15} "
                  f"AdvInf: {adversarial_influence:.2f}")

        # Check termination conditions
        if resources_after < sim_config.resources.threshold or not transform_success:
            termination_reason = "transform_failure" if not transform_success else "resource_threshold"
            print(f"  [PID {worker_pid}] TERMINATED: {termination_reason} at R{actual_round_completed+1}")
            break
            
    return pd.DataFrame(timeline_data_list)



def run_simulation_task(run_params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    ENHANCED SIMULATION TASK: Timeline-enabled simulation execution with streamlined metadata.
    
    ARCHITECTURAL INTEGRATION:
    - Maintain existing interface for backward compatibility
    - Add timeline data to returned DataFrame  
    - Preserve essential metadata structure
    - Enable timeline analysis capabilities
    
    ERROR HANDLING STRATEGY:
    - Graceful degradation for configuration errors
    - Structured error reporting in timeline format
    - Preserve debugging information
    - Maintain data consistency under failure conditions
    
    PERFORMANCE CONSIDERATIONS:
    - Minimal overhead addition to existing logic
    - Efficient DataFrame construction
    - Optimized data types for memory usage
    - Streamlined error path execution
    """
    worker_pid = os.getpid()
    worker_start_time = time.time()
    run_id = run_params['run_id']

    print(f"[PID {worker_pid}, RunID {run_id}] Starting timeline simulation. Factory: {run_params['config_factory_name']}")

    try:
        # CONFIGURATION RESOLUTION
        factory_name = run_params['config_factory_name']
        config_factory = CONFIG_FACTORIES.get(factory_name)

        if not config_factory:
            raise ValueError(f"Unknown config_factory_name: {factory_name}")

        # FACTORY ARGUMENT PREPARATION
        factory_args = {
            'mechanism': run_params['mechanism'],
            'adversarial_proportion_total': run_params['adversarial_proportion_total'],
            'seed': run_params['unique_config_seed'],
        }
        sim_config = config_factory(**factory_args)
        
        # JAX RANDOM KEY GENERATION
        key = jr.PRNGKey(run_params['unique_config_seed']) 
        
        # LLM SERVICE INITIALIZATION (optional)
        llm_service: Optional[ProcessIsolatedLLMService] = None
        if run_params.get('llm_model'):
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_api_key:
                try:
                    llm_service = ProcessIsolatedLLMService(
                        model=run_params['llm_model'],
                        api_key=openrouter_api_key,
                        process_id=f"{worker_pid}-{run_id}"
                    )
                    print(f"[PID {worker_pid}, RunID {run_id}] LLM service initialized: {run_params['llm_model']}")
                except Exception as e_llm:
                    print(f"[PID {worker_pid}, RunID {run_id}] LLM init failed: {e_llm}")
                    llm_service = None
            else:
                print(f"[PID {worker_pid}, RunID {run_id}] No OPENROUTER_API_KEY, running without LLM")
        
        # CORE SIMULATION EXECUTION
        results_df = _execute_single_simulation_logic(key, sim_config, llm_service, worker_pid, run_id)
        
        # TIMELINE METADATA ATTACHMENT: Add run identification to each timeline row
        timeline_metadata_columns = {
            'run_id': run_params['run_id'],
            'experiment_name': run_params['experiment_name'],
            'mechanism': run_params['mechanism'],
            'adversarial_proportion_total': run_params['adversarial_proportion_total'],
            'replication_run_index': run_params['replication_run_index'],
            'unique_config_seed': run_params['unique_config_seed'],
            'num_crops_config': len(sim_config.crops),
            'llm_model': run_params.get('llm_model', 'none')
        }
        
        # ADD METADATA TO EACH TIMELINE ROW
        for col_name, col_val in timeline_metadata_columns.items():
            if col_name not in results_df.columns:
                results_df[col_name] = col_val

        # AGGREGATED METADATA for results system compatibility
        final_resources = float(results_df['resources_after'].iloc[-1]) if not results_df.empty else 0.0
        aggregated_metadata = {
            **run_params,
            'status': 'success',
            'worker_pid': worker_pid,
            'final_resources': final_resources,
            'rounds_completed': len(results_df) if not results_df.empty else 0,
            'simulation_duration_sec': time.time() - worker_start_time,
            'llm_actually_used': llm_service is not None
        }
        
        print(f"[PID {worker_pid}, RunID {run_id}] Success. Duration: {aggregated_metadata['simulation_duration_sec']:.2f}s, "
              f"Timeline Points: {len(results_df)}, Final Resources: {final_resources:.2f}")
        
        return results_df, aggregated_metadata

    except Exception as e_task:
        # ERROR HANDLING: Structured error reporting
        error_duration = time.time() - worker_start_time
        error_tb = traceback.format_exc()
        print(f"[PID {worker_pid}, RunID {run_id}] FAILED: {e_task}\n{error_tb}")
        
        # STRUCTURED ERROR METADATA
        error_metadata = {
            **run_params,
            'status': 'error',
            'worker_pid': worker_pid,
            'error_message': str(e_task),
            'error_traceback': error_tb,
            'simulation_duration_sec': error_duration,
            'final_resources': 0.0,
            'rounds_completed': 0,
            'llm_actually_used': False
        }
        
        # ERROR TIMELINE DATA (single row indicating failure)
        error_timeline_data = {
            'round': 0,
            'resources_after': 0.0,
            'resource_change': 0.0,
            'decision_idx': -1,
            'chosen_portfolio': 'ERROR',
            'transform_success': False,
            'error': str(e_task),
            **{k: run_params.get(k) for k in ['run_id', 'experiment_name', 'mechanism', 'adversarial_proportion_total']}
        }
        
        return pd.DataFrame([error_timeline_data]), error_metadata