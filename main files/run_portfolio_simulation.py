# main_simulation_runner.py

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Literal, Optional, Dict, Any
import os # For potential API key loading

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Project-specific imports
from core.graph import GraphState
from core.category import Transform # Assuming this is a Callable[[GraphState], GraphState]

from environments.democracy.configuration import (
    PortfolioDemocracyConfig,
    create_thesis_baseline_config # Factory to create specific experimental configs
)
from environments.democracy.initialization import (
    initialize_portfolio_democracy_graph_state,
    get_true_expected_yields_for_round # Helper for updating yields per round
)
# Use the user-provided mechanism_factory
from environments.democracy.mechanism_factory import create_portfolio_mechanism_pipeline

from services.llm import LLMService, create_llm_service # For LLM integration

from execution.instrumentation.metrics import MetricsCollector
from execution.instrumentation.history import HistoryTracker # For more detailed history if needed

# --- Simulation Execution Function (minor change to pass llm_service) ---
def run_single_simulation(
    key: jr.PRNGKey,
    sim_config: PortfolioDemocracyConfig,
    llm_service: Optional[LLMService] # Explicitly pass LLM service
) -> pd.DataFrame:
    # ... (initialization and instrumentation setup largely the same) ...
    initial_state = initialize_portfolio_democracy_graph_state(key, sim_config)

    # Create the per-round transformation pipeline
    # NOW, pass the llm_service to the factory
    round_transform = create_portfolio_mechanism_pipeline(
        mechanism=sim_config.mechanism,
        llm_service=llm_service, # Pass the instantiated LLM service
        sim_config=sim_config
    )
    # ... (rest of the simulation loop and metrics collection) ...
    # ... (inside the loop, current_global_attrs["round_num"] is already set by housekeeping_transform) ...

    metrics_collector = MetricsCollector(metrics=["time", "resources", "decision_quality", "adversarial_impact"])
    per_round_summary_log = []
    current_state = initial_state

    print(f"\n--- Running Simulation: Mechanism={sim_config.mechanism}, "
          f"AdvTotal={sim_config.agent_settings.adversarial_proportion_total:.1f}, "
          f"Seed={sim_config.seed} ---")

    for round_idx_loop in range(sim_config.num_rounds): # loop from 0 to num_rounds-1
        state_before_transform = current_state # For metrics
        
        # Housekeeping (like round number update) is now the first step in round_transform
        # Ensure true expected yields are updated based on the *new* round number
        # The housekeeping transform will increment round_num. So, we might need to
        # get yields *after* housekeeping, or pass round_idx_loop to yield generator.
        # Let's assume round_num in global_attrs is correctly managed by the pipeline's first step.

        # For clarity, let's update true yields for the *upcoming* round before calling round_transform
        # The housekeeping transform will increment round_num from its current value.
        # So if current_state.global_attrs['round_num'] is N, housekeeping makes it N+1.
        # We need yields for round N+1.
        
        # Let's refine: housekeeping will set round_num to current_round_idx_loop.
        # So after housekeeping, state.global_attrs['round_num'] IS the current round.
        
        # Temp state update for true yields, if not handled within pipeline start
        temp_global_attrs = dict(current_state.global_attrs)
        # The round_num that _prediction_market_signal_generator and _actual_yield_sampling_transform use
        # will be the one *after* housekeeping_transform has run within the `round_transform` pipeline.
        # So, we fetch yields for the current state's round number
        # (which will be incremented by housekeeping at the start of round_transform)
        temp_global_attrs["current_true_expected_crop_yields"] = get_true_expected_yields_for_round(
            temp_global_attrs.get("round_num", -1) +1, # For the round about to be processed
            sim_config.crops
        )
        current_state_with_updated_yields = current_state.replace(global_attrs=temp_global_attrs)


        metrics_collector.start_collection()
        # The round_transform now includes housekeeping as its first step
        next_state = round_transform(current_state_with_updated_yields) 
        metrics_collector.finish_collection(round_transform, current_state_with_updated_yields, next_state)
        
        current_state = next_state
        
        # Correct round number for logging from the state AFTER the transform
        actual_round_logged = current_state.global_attrs.get("round_num", round_idx_loop)

        log_entry = {
            "round": actual_round_logged, # Use round from state
            "total_resources": current_state.global_attrs["current_total_resources"],
            "decision_idx": current_state.global_attrs.get("current_decision", -1),
        }
        # ... (rest of logging and termination)
        if log_entry["decision_idx"] != -1 and log_entry["decision_idx"] < len(sim_config.portfolios):
            log_entry["chosen_portfolio_name"] = sim_config.portfolios[log_entry["decision_idx"]].name
        else:
            log_entry["chosen_portfolio_name"] = "N/A"
        per_round_summary_log.append(log_entry)

        if (actual_round_logged +1 ) % 10 == 0 or actual_round_logged == sim_config.num_rounds -1 :
            print(f"  Round {actual_round_logged+1}/{sim_config.num_rounds} | "
                  f"Resources: {log_entry['total_resources']:.2f} | "
                  f"Decision: {log_entry['chosen_portfolio_name']} (idx {log_entry['decision_idx']})")

        if current_state.global_attrs["current_total_resources"] < sim_config.resources.threshold:
            print(f"  Simulation terminated early at round {actual_round_logged+1}: Resources ({log_entry['total_resources']:.2f}) "
                  f"below threshold ({sim_config.resources.threshold}).")
            # Fill remaining rounds in metrics_collector
            # Ensure the 'round' in these filled entries is also incremented
            last_metrics_entry = metrics_collector.metrics_history[-1].copy()
            for r_fill in range(actual_round_logged + 1, sim_config.num_rounds):
                fill_entry = last_metrics_entry.copy()
                fill_entry["round"] = r_fill 
                metrics_collector.metrics_history.append(fill_entry)
            break
            
    print(f"--- Simulation Ended ---")
    return metrics_collector.get_dataframe()


# --- Main Execution Block (Instantiate LLM Service) ---
if __name__ == "__main__":
    master_key = jr.PRNGKey(2025) 

    # --- LLM Service Setup ---
    try:
        # Ensure OPENROUTER_API_KEY is set in your environment
        llm_api_key = os.getenv("OPENROUTER_API_KEY")
        if not llm_api_key:
            print("Warning: OPENROUTER_API_KEY not found. LLM calls will fail if service is used.")
            llm_service = None
        else:
            # Using a cheap and relatively fast model. DeepSeek Coder is good.
            # You can also try "mistralai/mistral-7b-instruct" or "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
            # For testing, even very small models like "gryphlabs/gryphe-phi-2" might work for structure.
            llm_service = create_llm_service(model="deepseek/deepseek-coder", api_key=llm_api_key)
            # Test the service with a dummy call (optional)
            # try:
            #     print("Testing LLM Service...")
            #     test_resp = llm_service.generate("Hello!", max_tokens=5)
            #     print(f"LLM Service Test Response: {test_resp}")
            #     print("LLM Service Initialized and Tested.")
            # except Exception as e:
            #     print(f"LLM Service test failed: {e}")
            #     llm_service = None # Fallback if test fails
            print("LLM Service Initialized.")

    except Exception as e:
        print(f"Error initializing LLM Service: {e}")
        llm_service = None
    
    if not llm_service:
        print("LLM Service NOT initialized or failed. Running with placeholder agent logic if any.")
    
    # ... (rest of the experimental setup and loops, unchanged from your previous version) ...
    mechanisms_to_test: List[Literal["PDD", "PRD", "PLD"]] = ["PDD"] # Test with one first
    adversarial_proportions_total_range = [0.0, 0.2]  # Smaller range for testing
    adversarial_delegate_proportion_fixed = 0.25
    prediction_market_sigma_fixed = 0.25
    num_simulation_replications = 1 # Test with 1 replication first

    all_experiment_results_dfs = [] 
    experiment_seed_counter = 0 

    for mechanism in mechanisms_to_test:
        for adv_prop_total in adversarial_proportions_total_range:
            print(f"\n\n{'='*10} Starting Experimental Cell: Mechanism={mechanism}, AdvTotalProp={adv_prop_total:.1f} {'='*10}")
            cell_metrics_dfs = []
            for i_replication in range(num_simulation_replications):
                replication_key, master_key = jr.split(master_key) 
                current_config_seed = experiment_seed_counter + i_replication
                sim_config = create_thesis_baseline_config(
                    mechanism=mechanism,
                    adversarial_proportion_total=adv_prop_total,
                    adversarial_proportion_delegates=adversarial_delegate_proportion_fixed,
                    prediction_market_sigma=prediction_market_sigma_fixed,
                    seed=current_config_seed 
                )
                # Pass the llm_service to run_single_simulation
                replication_metrics_df = run_single_simulation(replication_key, sim_config, llm_service)
                
                replication_metrics_df['mechanism'] = mechanism
                replication_metrics_df['adv_prop_total'] = adv_prop_total
                replication_metrics_df['pm_sigma'] = prediction_market_sigma_fixed
                replication_metrics_df['config_seed'] = current_config_seed
                replication_metrics_df['replication_run'] = i_replication
                cell_metrics_dfs.append(replication_metrics_df)
            
            if cell_metrics_dfs:
                all_experiment_results_dfs.extend(cell_metrics_dfs)
            experiment_seed_counter += num_simulation_replications 

    if all_experiment_results_dfs:
        final_results_df = pd.concat(all_experiment_results_dfs, ignore_index=True)
        final_results_df.to_csv("simulation_results_llm_raw.csv", index=False)
        print("\n\n--- All Experiments Completed ---")
        print("Raw results saved to simulation_results_llm_raw.csv")
        # ... (plotting logic) ...
        # Plot: Average Final Resources vs. Adversarial Proportion (for fixed PM sigma)
        if not final_results_df.empty:
            idx = final_results_df.groupby(['mechanism', 'adv_prop_total', 'pm_sigma', 'config_seed', 'replication_run'])['round'].idxmax()
            final_round_data = final_results_df.loc[idx]
            avg_final_resources = final_round_data.groupby(
                ['mechanism', 'adv_prop_total', 'pm_sigma']
            )['resources_after'].mean().reset_index()

            plt.figure(figsize=(12, 7))
            for mechanism_type_plot in mechanisms_to_test: # Ensure you iterate over the tested ones
                plot_data = avg_final_resources[
                    (avg_final_resources['mechanism'] == mechanism_type_plot) &
                    (avg_final_resources['pm_sigma'] == prediction_market_sigma_fixed)
                ]
                if not plot_data.empty:
                    plt.plot(plot_data['adv_prop_total'], plot_data['resources_after'], marker='o', linestyle='-', label=mechanism_type_plot)
            
            plt.title(f'Avg. Final Resources vs. Adversarial Proportion (PM Sigma={prediction_market_sigma_fixed})')
            plt.xlabel('Total Adversarial Proportion (M)')
            plt.ylabel('Average Final Resources (End of Sim / Collapse)')
            # Get threshold from a sample config for plotting
            sample_thr_config = create_thesis_baseline_config("PDD")
            plt.axhline(y=sample_thr_config.resources.threshold, color='r', linestyle='--', label='Survival Threshold')
            plt.legend()
            plt.grid(True, alpha=0.7)
            plt.savefig("plot_avg_final_resources_llm.png")
            print("\nPlot saved: plot_avg_final_resources_llm.png")

            final_round_data['survived'] = final_round_data['resources_after'] >= sample_thr_config.resources.threshold
            survival_rate = final_round_data.groupby(
                 ['mechanism', 'adv_prop_total', 'pm_sigma']
            )['survived'].mean().reset_index()
            survival_rate.rename(columns={'survived': 'survival_rate'}, inplace=True)

            plt.figure(figsize=(12, 7))
            for mechanism_type_plot in mechanisms_to_test:
                plot_data = survival_rate[
                    (survival_rate['mechanism'] == mechanism_type_plot) &
                    (survival_rate['pm_sigma'] == prediction_market_sigma_fixed)
                ]
                if not plot_data.empty:
                    plt.plot(plot_data['adv_prop_total'], plot_data['survival_rate'], marker='x', linestyle='-', label=mechanism_type_plot)

            plt.title(f'Survival Rate vs. Adversarial Proportion (PM Sigma={prediction_market_sigma_fixed})')
            plt.xlabel('Total Adversarial Proportion (M)')
            plt.ylabel('Survival Rate (Fraction of Sims > Threshold)')
            plt.ylim(-0.05, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.7)
            plt.savefig("plot_survival_rate_llm.png")
            print("Plot saved: plot_survival_rate_llm.png")
    else:
        print("No simulation results were generated to analyze.")