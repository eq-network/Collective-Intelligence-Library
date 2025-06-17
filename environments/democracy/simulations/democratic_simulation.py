# main_simulation_runner.py

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Literal, Optional, Dict, Any, Tuple
import os
import time
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Project-specific imports
from algebra.graph import GraphState
from algebra.category import Transform

from environments.random.configuration import (
    PortfolioDemocracyConfig,
    create_thesis_baseline_config
)
from environments.random.initialization import (
    initialize_portfolio_democracy_graph_state,
    get_true_expected_yields_for_round
)
from environments.random.mechanism_factory import create_portfolio_mechanism_pipeline

from services.llm import LLMService, create_llm_service

# --- Worker Function for Parallel Execution ---
def run_single_simulation_worker(params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enhanced worker function with complete process isolation and error tracking.
    """
    process_id = os.getpid()
    worker_start_time = time.time()
    
    try:
        # Extract parameters with validation
        required_params = ['mechanism', 'adv_prop_total', 'config_seed', 'replication_run']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        mechanism = params['mechanism']
        adv_prop_total = params['adv_prop_total']
        config_seed = params['config_seed']
        replication_run = params['replication_run']
        adversarial_delegate_proportion = params.get('adversarial_delegate_proportion', 0.25)
        prediction_market_sigma = params.get('prediction_market_sigma', 0.25)
        llm_model = params.get('llm_model', 'deepseek/deepseek-coder')
        
        print(f"[PID {process_id}] Starting simulation: {mechanism}-{adv_prop_total}-Rep{replication_run}")
        
        # Initialize process-isolated LLM service
        llm_service = None
        llm_init_start = time.time()
        
        try:
            llm_api_key = os.getenv("OPENROUTER_API_KEY")
            if llm_api_key:
                llm_service = ProcessIsolatedLLMService(
                    model=llm_model,
                    api_key=llm_api_key,
                    process_id=str(process_id)
                )
                print(f"[PID {process_id}] LLM service initialized in {time.time() - llm_init_start:.2f}s")
            else:
                print(f"[PID {process_id}] No API key available, running without LLM")
                
        except Exception as e:
            print(f"[PID {process_id}] LLM initialization failed: {e}")
            llm_service = None
        
        # Create simulation configuration with process-specific seed
        sim_config = create_thesis_baseline_config(
            mechanism=mechanism,
            adversarial_proportion_total=adv_prop_total,
            adversarial_proportion_delegates=adversarial_delegate_proportion,
            prediction_market_sigma=prediction_market_sigma,
            seed=config_seed + process_id  # Add process ID to ensure unique seeds
        )
        
        # Generate JAX key with process-specific entropy
        key = jr.PRNGKey(config_seed + process_id + int(time.time() * 1000) % 10000)
        
        # Run the actual simulation
        sim_start_time = time.time()
        result_df = run_single_simulation_core(key, sim_config, llm_service, process_id)
        sim_duration = time.time() - sim_start_time
        
        # Add comprehensive metadata
        result_df['mechanism'] = mechanism
        result_df['adv_prop_total'] = adv_prop_total
        result_df['pm_sigma'] = prediction_market_sigma
        result_df['config_seed'] = config_seed
        result_df['replication_run'] = replication_run
        result_df['worker_pid'] = process_id
        result_df['simulation_duration'] = sim_duration
        
        # Create detailed metadata
        metadata = {
            'mechanism': mechanism,
            'adv_prop_total': adv_prop_total,
            'config_seed': config_seed,
            'replication_run': replication_run,
            'status': 'success',
            'worker_pid': process_id,
            'final_resources': float(result_df['resources_after'].iloc[-1]) if len(result_df) > 0 else 0.0,
            'rounds_completed': len(result_df),
            'simulation_duration': sim_duration,
            'total_worker_duration': time.time() - worker_start_time,
            'llm_service_available': llm_service is not None
        }
        
        print(f"[PID {process_id}] Simulation completed successfully in {sim_duration:.2f}s")
        return result_df, metadata
        
    except Exception as e:
        error_duration = time.time() - worker_start_time
        
        # Comprehensive error metadata
        error_metadata = {
            'mechanism': params.get('mechanism', 'unknown'),
            'adv_prop_total': params.get('adv_prop_total', 0.0),
            'config_seed': params.get('config_seed', 0),
            'replication_run': params.get('replication_run', 0),
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__,
            'error_traceback': traceback.format_exc(),
            'worker_pid': process_id,
            'error_duration': error_duration
        }
        
        print(f"[PID {process_id}] Simulation failed after {error_duration:.2f}s: {e}")
        
        # Return empty dataframe with error information
        empty_df = pd.DataFrame([{
            'round': 0,
            'resources_after': 0.0,
            'mechanism': params.get('mechanism', 'unknown'),
            'adv_prop_total': params.get('adv_prop_total', 0.0),
            'config_seed': params.get('config_seed', 0),
            'replication_run': params.get('replication_run', 0),
            'error': str(e),
            'worker_pid': process_id
        }])
        
        return empty_df, error_metadata


# --- Core Simulation Logic (Extracted for Worker) ---
def run_single_simulation_core(
    key: jr.PRNGKey,
    sim_config: PortfolioDemocracyConfig,
    llm_service: Optional[ProcessIsolatedLLMService],
    process_id: int
) -> pd.DataFrame:
    """
    Core simulation execution with enhanced process tracking and error handling.
    """
    # Initialize state
    initial_state = initialize_portfolio_democracy_graph_state(key, sim_config)
    
    # Create transformation pipeline
    round_transform = create_portfolio_mechanism_pipeline(
        mechanism=sim_config.mechanism,
        llm_service=llm_service._service if llm_service else None,  # Extract underlying service
        sim_config=sim_config
    )
    
    # Initialize data collection with process tracking
    simulation_data = []
    current_state = initial_state
    
    print(f"[PID {process_id}] Running Simulation: Mechanism={sim_config.mechanism}, "
          f"AdvTotalProp={sim_config.agent_settings.adversarial_proportion_total:.1f}, "
          f"Seed={sim_config.seed}")
    
    # Run simulation for specified number of rounds
    for round_idx_loop in range(sim_config.num_rounds):
        round_start_time = time.time()
        
        # Track pre-transformation state
        resources_before = current_state.global_attrs.get("current_total_resources", 0)
        
        # Update expected yields for upcoming round
        temp_global_attrs = dict(current_state.global_attrs)
        temp_global_attrs["current_true_expected_crop_yields"] = get_true_expected_yields_for_round(
            temp_global_attrs.get("round_num", -1) + 1,
            sim_config.crops
        )
        current_state_with_updated_yields = current_state.replace(global_attrs=temp_global_attrs)
        
        # Apply transformation with error handling
        try:
            next_state = round_transform(current_state_with_updated_yields)
            transform_success = True
        except Exception as e:
            print(f"[PID {process_id}] Error in round {round_idx_loop}: {e}")
            next_state = current_state_with_updated_yields  # Use previous state
            transform_success = False
        
        execution_time = time.time() - round_start_time
        
        # Capture post-transformation state
        current_state = next_state
        resources_after = current_state.global_attrs.get("current_total_resources", 0)
        
        # Calculate resource changes
        resource_change = resources_after - resources_before
        resource_change_pct = 0
        if resources_before > 0:
            resource_change_pct = ((resources_after / resources_before) - 1) * 100
        
        # Get round information
        actual_round = current_state.global_attrs.get("round_num", round_idx_loop)
        decision_idx = current_state.global_attrs.get("current_decision", -1)
        
        # Capture adversarial influence
        adversarial_influence = 0.0
        if "is_adversarial" in current_state.node_attrs and "voting_power" in current_state.node_attrs:
            adversarial = current_state.node_attrs["is_adversarial"]
            voting_power = current_state.node_attrs["voting_power"]
            total_power = jnp.sum(voting_power)
            adversarial_power = jnp.sum(voting_power * adversarial)
            if total_power > 0:
                adversarial_influence = float(adversarial_power / total_power)
        
        # Compile round data with process information
        round_data = {
            "round": actual_round,
            "execution_time": execution_time,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "resource_change": resource_change,
            "resource_change_pct": resource_change_pct,
            "adversarial_influence": adversarial_influence,
            "decision_idx": decision_idx,
            "transform_success": transform_success,
            "process_id": process_id
        }
        
        # Add portfolio name if available
        if decision_idx != -1 and decision_idx < len(sim_config.portfolios):
            round_data["chosen_portfolio"] = sim_config.portfolios[decision_idx].name
        else:
            round_data["chosen_portfolio"] = "N/A"
        
        # Store round data
        simulation_data.append(round_data)
        
        # Print progress at intervals (reduced frequency for parallel execution)
        if (actual_round + 1) % 20 == 0 or actual_round == sim_config.num_rounds - 1:
            print(f"  [PID {process_id}] Round {actual_round+1}/{sim_config.num_rounds} | "
                  f"Resources: {resources_after:.2f} | "
                  f"Decision: {round_data['chosen_portfolio']} (idx {decision_idx}) | "
                  f"Success: {transform_success}")
        
        # Check termination condition
        if current_state.global_attrs["current_total_resources"] < sim_config.resources.threshold:
            print(f"  [PID {process_id}] Simulation terminated early at round {actual_round+1}: "
                  f"Resources ({resources_after:.2f}) below threshold "
                  f"({sim_config.resources.threshold}).")
            
            # Fill remaining rounds with last state data for consistency
            last_round_data = dict(round_data)
            for r_fill in range(actual_round + 1, sim_config.num_rounds):
                fill_data = dict(last_round_data)
                fill_data["round"] = r_fill
                fill_data["execution_time"] = 0  # No execution for filled rounds
                fill_data["transform_success"] = False  # No actual transformation
                simulation_data.append(fill_data)
            break
    
    print(f"[PID {process_id}] Simulation Completed")
    
    # Convert collected data to DataFrame
    return pd.DataFrame(simulation_data)

# --- Main Execution Block (Enhanced with Parallelization) ---
if __name__ == "__main__":
    # Set master random key
    master_key = jr.PRNGKey(2025)
    
    # Record start time for overall execution
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- Experimental Setup (Reduced Complexity) ---
    # Test all three mechanisms
    mechanisms_to_test: List[Literal["PDD", "PRD", "PLD"]] = ["PDD", "PRD", "PLD"]
    
    # REDUCED: Test fewer adversarial proportions for "less comprehensive" testing
    adversarial_proportions_total_range = [0.1, 0.25, 0.4]  # Reduced from 5 to 3 values
    
    # Fixed parameters
    adversarial_delegate_proportion_fixed = 0.25
    prediction_market_sigma_fixed = 0.25
    
    # Fixed replications (already 3 as requested)
    num_simulation_replications = 3
    
    # Parallelization settings
    max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers to avoid overwhelming system
    
    print(f"\n===== STARTING STREAMLINED PARALLEL TESTING =====")
    print(f"Mechanisms: {mechanisms_to_test}")
    print(f"Adversarial Proportions: {adversarial_proportions_total_range}")
    print(f"Replications per configuration: {num_simulation_replications}")
    print(f"Agent Configuration: 15 agents, 6 delegates, 50 rounds")
    print(f"Parallel Workers: {max_workers}")
    
    # Calculate total experiments
    total_experiments = len(mechanisms_to_test) * len(adversarial_proportions_total_range) * num_simulation_replications
    print(f"Total planned simulations: {total_experiments}")
    print(f"=======================================\n")
    
    # --- Build Parameter Grid for Parallel Execution ---
    param_grid = []
    experiment_seed_counter = 0
    
    for mechanism in mechanisms_to_test:
        for adv_prop_total in adversarial_proportions_total_range:
            for i_replication in range(num_simulation_replications):
                current_config_seed = experiment_seed_counter + i_replication
                
                param_grid.append({
                    'mechanism': mechanism,
                    'adv_prop_total': adv_prop_total,
                    'adversarial_delegate_proportion': adversarial_delegate_proportion_fixed,
                    'prediction_market_sigma': prediction_market_sigma_fixed,
                    'config_seed': current_config_seed,
                    'replication_run': i_replication,
                    'llm_model': 'deepseek/deepseek-coder'
                })
            
            experiment_seed_counter += num_simulation_replications
    
    # --- Execute Simulations in Parallel ---
    all_experiment_results = []
    all_metadata = []
    completed_experiments = 0
    
    print(f"Starting parallel execution with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {
            executor.submit(run_single_simulation_worker, params): params 
            for params in param_grid
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            completed_experiments += 1
            progress_pct = (completed_experiments / total_experiments) * 100
            
            try:
                result_df, metadata = future.result()
                
                if metadata['status'] == 'success':
                    all_experiment_results.append(result_df)
                    print(f"✓ Completed [{completed_experiments}/{total_experiments}] "
                          f"{params['mechanism']}-{params['adv_prop_total']:.1f}-Rep{params['replication_run']} "
                          f"({progress_pct:.1f}%) - Final Resources: {metadata['final_resources']:.2f}")
                else:
                    print(f"✗ Failed [{completed_experiments}/{total_experiments}] "
                          f"{params['mechanism']}-{params['adv_prop_total']:.1f}-Rep{params['replication_run']} "
                          f"({progress_pct:.1f}%) - Error: {metadata['error_message']}")
                
                all_metadata.append(metadata)
                
            except Exception as e:
                print(f"✗ Unexpected error [{completed_experiments}/{total_experiments}] "
                      f"{params['mechanism']}-{params['adv_prop_total']:.1f}-Rep{params['replication_run']}: {e}")
    
    # --- Process and Save Results ---
    if all_experiment_results:
        # Combine all results into one DataFrame
        final_results_df = pd.concat(all_experiment_results, ignore_index=True)
        
        # Save raw results
        results_filename = f"simulation_results_parallel_{timestamp}.csv"
        final_results_df.to_csv(results_filename, index=False)
        print(f"\n--- All Parallel Experiments Completed ---")
        print(f"Raw results saved to {results_filename}")
        
        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_filename = f"simulation_metadata_{timestamp}.csv"
        metadata_df.to_csv(metadata_filename, index=False)
        print(f"Execution metadata saved to {metadata_filename}")
        
        # Print success/failure summary
        success_count = len(metadata_df[metadata_df['status'] == 'success'])
        failure_count = len(metadata_df[metadata_df['status'] == 'error'])
        print(f"Execution Summary: {success_count} successful, {failure_count} failed")
        
        # --- Generate Enhanced Visualizations ---
        # Filter out failed experiments for analysis
        valid_results = final_results_df[~final_results_df.get('error', pd.Series([False]*len(final_results_df))).notna()]
        
        if len(valid_results) > 0:
            # Get last round data for final resource comparisons
            idx = valid_results.groupby(['mechanism', 'adv_prop_total', 'pm_sigma', 'config_seed', 'replication_run'])['round'].idxmax()
            final_round_data = valid_results.loc[idx]
            
            # Calculate average final resources by mechanism and adversarial proportion
            avg_final_resources = final_round_data.groupby(
                ['mechanism', 'adv_prop_total', 'pm_sigma']
            )['resources_after'].mean().reset_index()
            
            # Calculate standard deviation for error bars
            std_final_resources = final_round_data.groupby(
                ['mechanism', 'adv_prop_total', 'pm_sigma']
            )['resources_after'].std().reset_index().rename(columns={'resources_after': 'resources_std'})
            
            # Merge mean and std
            avg_final_resources = pd.merge(avg_final_resources, std_final_resources, 
                                          on=['mechanism', 'adv_prop_total', 'pm_sigma'])
            
            # Plot 1: Average Final Resources vs. Adversarial Proportion
            plt.figure(figsize=(12, 8))
            
            colors = {'PDD': 'blue', 'PRD': 'green', 'PLD': 'red'}
            markers = {'PDD': 'o', 'PRD': 's', 'PLD': '^'}
            
            for mechanism_type_plot in mechanisms_to_test:
                plot_data = avg_final_resources[
                    (avg_final_resources['mechanism'] == mechanism_type_plot) &
                    (avg_final_resources['pm_sigma'] == prediction_market_sigma_fixed)
                ]
                if not plot_data.empty:
                    plt.errorbar(
                        plot_data['adv_prop_total'], 
                        plot_data['resources_after'],
                        yerr=plot_data['resources_std'],
                        marker=markers.get(mechanism_type_plot, 'o'),
                        linestyle='-',
                        color=colors.get(mechanism_type_plot, 'black'),
                        label=f"{mechanism_type_plot}",
                        capsize=5
                    )
            
            plt.title(f'Avg. Final Resources vs. Adversarial Proportion\n(15 Agents, PM Sigma={prediction_market_sigma_fixed})')
            plt.xlabel('Adversarial Agent Proportion')
            plt.ylabel('Average Final Resources')
            plt.grid(True, alpha=0.3)
            
            # Add threshold line
            sample_thr_config = create_thesis_baseline_config("PDD")
            plt.axhline(y=sample_thr_config.resources.threshold, color='r', linestyle='--', 
                       label=f'Survival Threshold ({sample_thr_config.resources.threshold})')
            
            # Improve x-axis
            plt.xticks(adversarial_proportions_total_range, 
                      [f"{int(x*100)}%" for x in adversarial_proportions_total_range])
            plt.xlim(0.05, 0.45)
            
            plt.legend(loc='best')
            resources_plot_filename = f"plot_avg_final_resources_parallel_{timestamp}.png"
            plt.savefig(resources_plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {resources_plot_filename}")
            
            # Plot 2: Survival Rate
            final_round_data['survived'] = final_round_data['resources_after'] >= sample_thr_config.resources.threshold
            survival_rate = final_round_data.groupby(
                ['mechanism', 'adv_prop_total', 'pm_sigma']
            )['survived'].mean().reset_index()
            survival_rate.rename(columns={'survived': 'survival_rate'}, inplace=True)
            
            plt.figure(figsize=(12, 8))
            
            for mechanism_type_plot in mechanisms_to_test:
                plot_data = survival_rate[
                    (survival_rate['mechanism'] == mechanism_type_plot) &
                    (survival_rate['pm_sigma'] == prediction_market_sigma_fixed)
                ]
                if not plot_data.empty:
                    plt.plot(
                        plot_data['adv_prop_total'], 
                        plot_data['survival_rate'],
                        marker=markers.get(mechanism_type_plot, 'x'),
                        linestyle='-',
                        color=colors.get(mechanism_type_plot, 'black'),
                        linewidth=2,
                        label=mechanism_type_plot
                    )
            
            plt.title(f'Survival Rate vs. Adversarial Proportion\n(15 Agents, PM Sigma={prediction_market_sigma_fixed})')
            plt.xlabel('Adversarial Agent Proportion')
            plt.ylabel('Survival Rate')
            plt.ylim(-0.05, 1.05)
            plt.grid(True, alpha=0.3)
            
            plt.xticks(adversarial_proportions_total_range, 
                      [f"{int(x*100)}%" for x in adversarial_proportions_total_range])
            plt.xlim(0.05, 0.45)
            
            plt.legend(loc='best')
            survival_plot_filename = f"plot_survival_rate_parallel_{timestamp}.png"
            plt.savefig(survival_plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {survival_plot_filename}")
            
            # Create summary table
            summary_table = pd.DataFrame({
                'Mechanism': avg_final_resources['mechanism'],
                'Adversarial %': (avg_final_resources['adv_prop_total'] * 100).astype(int),
                'Final Resources': avg_final_resources['resources_after'].round(2),
                'Std Dev': avg_final_resources['resources_std'].round(2),
                'Survival Rate': survival_rate['survival_rate'].round(2)
            })
            
            # Save summary table
            summary_table_filename = f"mechanism_comparison_summary_parallel_{timestamp}.csv"
            summary_table.to_csv(summary_table_filename, index=False)
            print(f"Summary table saved: {summary_table_filename}")
            
        else:
            print("No valid simulation results were generated for analysis.")
    else:
        print("No simulation results were generated.")
    
    # Display total execution time
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal parallel execution time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    
    # Display performance improvement estimate
    estimated_sequential_time = total_duration * max_workers
    speedup = estimated_sequential_time / total_duration if total_duration > 0 else 1
    print(f"Estimated speedup: {speedup:.1f}x (compared to sequential execution)")