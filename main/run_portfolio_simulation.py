# main_simulation_runner.py

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Literal, Optional, Dict, Any
import os
import time
from datetime import datetime

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Project-specific imports
from core.graph import GraphState
from core.category import Transform

from environments.democracy.configuration import (
    PortfolioDemocracyConfig,
    create_thesis_baseline_config
)
from environments.democracy.initialization import (
    initialize_portfolio_democracy_graph_state,
    get_true_expected_yields_for_round
)
from environments.democracy.mechanism_factory import create_portfolio_mechanism_pipeline

from services.llm import LLMService, create_llm_service

# Note: MetricsCollector import is removed as we're not using it

# --- Modified Simulation Execution Function ---
def run_single_simulation(
    key: jr.PRNGKey,
    sim_config: PortfolioDemocracyConfig,
    llm_service: Optional[LLMService]
) -> pd.DataFrame:
    """
    Execute a single simulation run and return metrics in a DataFrame.
    
    Args:
        key: Random key for simulation
        sim_config: Configuration parameters
        llm_service: Optional LLM service for agent decision-making
        
    Returns:
        DataFrame containing simulation metrics
    """
    # Initialize state
    initial_state = initialize_portfolio_democracy_graph_state(key, sim_config)
    
    # Create transformation pipeline
    round_transform = create_portfolio_mechanism_pipeline(
        mechanism=sim_config.mechanism,
        llm_service=llm_service,
        sim_config=sim_config
    )
    
    # Initialize direct data collection
    simulation_data = []
    current_state = initial_state
    
    print(f"\n--- Running Simulation: Mechanism={sim_config.mechanism}, "
          f"AdvTotalProp={sim_config.agent_settings.adversarial_proportion_total:.1f}, "
          f"Seed={sim_config.seed} ---")
    
    # Run simulation for specified number of rounds
    for round_idx_loop in range(sim_config.num_rounds):
        # Track pre-transformation state
        resources_before = current_state.global_attrs.get("current_total_resources", 0)
        
        # Update expected yields for upcoming round
        temp_global_attrs = dict(current_state.global_attrs)
        temp_global_attrs["current_true_expected_crop_yields"] = get_true_expected_yields_for_round(
            temp_global_attrs.get("round_num", -1) + 1,
            sim_config.crops
        )
        current_state_with_updated_yields = current_state.replace(global_attrs=temp_global_attrs)
        
        # Measure execution time
        start_time = time.time()
        next_state = round_transform(current_state_with_updated_yields)
        execution_time = time.time() - start_time
        
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
        # (This was previously in MetricsCollector, we're now calculating it directly)
        adversarial_influence = 0.0
        if "is_adversarial" in current_state.node_attrs and "voting_power" in current_state.node_attrs:
            adversarial = current_state.node_attrs["is_adversarial"]
            voting_power = current_state.node_attrs["voting_power"]
            total_power = jnp.sum(voting_power)
            adversarial_power = jnp.sum(voting_power * adversarial)
            if total_power > 0:
                adversarial_influence = float(adversarial_power / total_power)
        
        # Compile round data
        round_data = {
            "round": actual_round,
            "execution_time": execution_time,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "resource_change": resource_change,
            "resource_change_pct": resource_change_pct,
            "adversarial_influence": adversarial_influence,
            "decision_idx": decision_idx,
        }
        
        # Add portfolio name if available
        if decision_idx != -1 and decision_idx < len(sim_config.portfolios):
            round_data["chosen_portfolio"] = sim_config.portfolios[decision_idx].name
        else:
            round_data["chosen_portfolio"] = "N/A"
        
        # Store round data
        simulation_data.append(round_data)
        
        # Print progress at intervals
        if (actual_round + 1) % 10 == 0 or actual_round == sim_config.num_rounds - 1:
            print(f"  Round {actual_round+1}/{sim_config.num_rounds} | "
                  f"Resources: {resources_after:.2f} | "
                  f"Decision: {round_data['chosen_portfolio']} (idx {decision_idx})")
        
        # Check termination condition
        if current_state.global_attrs["current_total_resources"] < sim_config.resources.threshold:
            print(f"  Simulation terminated early at round {actual_round+1}: "
                  f"Resources ({resources_after:.2f}) below threshold "
                  f"({sim_config.resources.threshold}).")
            
            # Fill remaining rounds with last state data for consistency
            last_round_data = dict(round_data)
            for r_fill in range(actual_round + 1, sim_config.num_rounds):
                fill_data = dict(last_round_data)
                fill_data["round"] = r_fill
                fill_data["execution_time"] = 0  # No execution for filled rounds
                simulation_data.append(fill_data)
            break
    
    print(f"--- Simulation Ended ---")
    
    # Convert collected data to DataFrame (replacing metrics_collector.get_dataframe())
    return pd.DataFrame(simulation_data)


# --- Main Execution Block (Modified for Comprehensive Testing) ---
if __name__ == "__main__":
    # Set master random key
    master_key = jr.PRNGKey(2025)
    
    # Record start time for overall execution
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- LLM Service Setup ---
    try:
        llm_api_key = os.getenv("OPENROUTER_API_KEY")
        if not llm_api_key:
            print("Warning: OPENROUTER_API_KEY not found. LLM calls will fail if service is used.")
            llm_service = None
        else:
            llm_service = create_llm_service(model="deepseek/deepseek-coder", api_key=llm_api_key)
            print("LLM Service Initialized.")
    except Exception as e:
        print(f"Error initializing LLM Service: {e}")
        llm_service = None
    
    if not llm_service:
        print("LLM Service NOT initialized or failed. Running with placeholder agent logic if any.")
    
    # --- Expanded Experimental Setup ---
    # Test all three mechanisms
    mechanisms_to_test: List[Literal["PDD", "PRD", "PLD"]] = ["PDD", "PRD", "PLD"]
    
    # Test adversarial proportions from 10% to 50%
    adversarial_proportions_total_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Fixed parameters
    adversarial_delegate_proportion_fixed = 0.25
    prediction_market_sigma_fixed = 0.25
    
    # Increase replications for statistical significance
    num_simulation_replications = 3
    
    # Container for all experiment results
    all_experiment_results = []
    experiment_seed_counter = 0
    
    # Calculate total experiments for progress tracking
    total_experiments = len(mechanisms_to_test) * len(adversarial_proportions_total_range) * num_simulation_replications
    completed_experiments = 0
    
    print(f"\n===== STARTING COMPREHENSIVE TESTING =====")
    print(f"Mechanisms: {mechanisms_to_test}")
    print(f"Adversarial Proportions: {adversarial_proportions_total_range}")
    print(f"Replications per configuration: {num_simulation_replications}")
    print(f"Total planned simulations: {total_experiments}")
    print(f"=======================================\n")
    
    # Run all experiments in nested loops
    for mechanism in mechanisms_to_test:
        mechanism_results = []
        
        for adv_prop_total in adversarial_proportions_total_range:
            print(f"\n\n{'='*20} Starting Experimental Cell: Mechanism={mechanism}, "
                  f"AdvTotalProp={adv_prop_total:.1f} {'='*20}")
            
            cell_results = []
            cell_start_time = time.time()
            
            for i_replication in range(num_simulation_replications):
                # Split random key for this replication
                replication_key, master_key = jr.split(master_key)
                current_config_seed = experiment_seed_counter + i_replication
                
                # Configure this simulation
                sim_config = create_thesis_baseline_config(
                    mechanism=mechanism,
                    adversarial_proportion_total=adv_prop_total,
                    adversarial_proportion_delegates=adversarial_delegate_proportion_fixed,
                    prediction_market_sigma=prediction_market_sigma_fixed,
                    seed=current_config_seed
                )
                
                # Run simulation and collect metrics
                replication_result_df = run_single_simulation(replication_key, sim_config, llm_service)
                
                # Add experimental condition metadata
                replication_result_df['mechanism'] = mechanism
                replication_result_df['adv_prop_total'] = adv_prop_total
                replication_result_df['pm_sigma'] = prediction_market_sigma_fixed
                replication_result_df['config_seed'] = current_config_seed
                replication_result_df['replication_run'] = i_replication
                
                # Save replication results
                cell_results.append(replication_result_df)
                
                # Update progress tracking
                completed_experiments += 1
                progress_pct = (completed_experiments / total_experiments) * 100
                print(f"  Completed replication {i_replication+1}/{num_simulation_replications} "
                      f"({completed_experiments}/{total_experiments} total, {progress_pct:.1f}% complete)")
            
            # Calculate and display cell completion statistics
            cell_duration = time.time() - cell_start_time
            print(f"Cell {mechanism}-{adv_prop_total:.1f} completed in {cell_duration:.1f} seconds")
            
            # Add cell results to mechanism results
            mechanism_results.extend(cell_results)
            
            # Increment seed counter
            experiment_seed_counter += num_simulation_replications
        
        # Add mechanism results to overall results
        all_experiment_results.extend(mechanism_results)
        
        # Save intermediate results for this mechanism (helps in case of failure)
        if mechanism_results:
            mechanism_df = pd.concat(mechanism_results, ignore_index=True)
            mechanism_filename = f"{mechanism}_results_{timestamp}.csv"
            mechanism_df.to_csv(mechanism_filename, index=False)
            print(f"Intermediate results for {mechanism} saved to {mechanism_filename}")
    
    # --- Process and save all results ---
    if all_experiment_results:
        # Combine all results into one DataFrame
        final_results_df = pd.concat(all_experiment_results, ignore_index=True)
        
        # Save raw results
        results_filename = f"simulation_results_{timestamp}.csv"
        final_results_df.to_csv(results_filename, index=False)
        print(f"\n\n--- All Experiments Completed ---")
        print(f"Raw results saved to {results_filename}")
        
        # --- Generate Enhanced Visualizations ---
        # Get last round data for final resource comparisons
        idx = final_results_df.groupby(['mechanism', 'adv_prop_total', 'pm_sigma', 'config_seed', 'replication_run'])['round'].idxmax()
        final_round_data = final_results_df.loc[idx]
        
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
        
        plt.title(f'Avg. Final Resources vs. Adversarial Proportion (PM Sigma={prediction_market_sigma_fixed})')
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
        plt.xlim(0.05, 0.55)  # Add some padding
        
        plt.legend(loc='best')
        resources_plot_filename = f"plot_avg_final_resources_{timestamp}.png"
        plt.savefig(resources_plot_filename)
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
        
        plt.title(f'Survival Rate vs. Adversarial Proportion (PM Sigma={prediction_market_sigma_fixed})')
        plt.xlabel('Adversarial Agent Proportion')
        plt.ylabel('Survival Rate')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        
        # Improve x-axis
        plt.xticks(adversarial_proportions_total_range, 
                  [f"{int(x*100)}%" for x in adversarial_proportions_total_range])
        plt.xlim(0.05, 0.55)  # Add some padding
        
        plt.legend(loc='best')
        survival_plot_filename = f"plot_survival_rate_{timestamp}.png"
        plt.savefig(survival_plot_filename)
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
        summary_table_filename = f"mechanism_comparison_summary_{timestamp}.csv"
        summary_table.to_csv(summary_table_filename, index=False)
        print(f"Summary table saved: {summary_table_filename}")
        
    else:
        print("No simulation results were generated to analyze.")
    
    # Display total execution time
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")