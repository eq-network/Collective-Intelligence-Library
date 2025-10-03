# main.py
"""
The main entry point for running all simulation experiments.

This script imports experiment definitions from specific applications
(e.g., democracy, epidemiology) and passes them to the generic
execution engine. This maintains a clean separation of concerns.
"""
import time
import os
import multiprocessing
import itertools
from datetime import datetime
from typing import List, Dict, Any
import importlib

# Generic execution components
from execution.runner import EnhancedParallelExperimentRunner
from execution.results import EnhancedResultsAggregator
from execution.analysis import run_full_analysis # Assume a general analysis script

def generate_run_list_from_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expands a list of experiment definitions into a flat list of single-run parameters
    that the generic worker can understand.
    """
    all_runs = []
    run_counter = 0

    for exp in experiments:
        sweep_keys = list(exp["config_sweeps"].keys())
        sweep_values = list(exp["config_sweeps"].values())
        
        for combo in itertools.product(*sweep_values):
            # Base parameters for a specific setting in the sweep
            run_base_params = exp["static_config_params"].copy()
            for i, key in enumerate(sweep_keys):
                run_base_params[key] = combo[i]
            
            # Create replications for this setting
            for rep in range(exp["num_replications_per_setting"]):
                run_params = {
                    "run_id": run_counter,
                    "experiment_name": exp['name'],
                    "replication_index": rep,
                    "unique_config_seed": exp['base_seed'] + run_counter,
                    "simulation_class_path": exp['simulation_class_path'],
                    # The worker will call this factory with the params below
                    "config_factory_path": exp['config_factory_path'],
                    "config_params": run_base_params.copy()
                }
                all_runs.append(run_params)
                run_counter += 1
                
    return all_runs

if __name__ == "__main__":
    multiprocessing.freeze_support()
    start_time = time.time()
    
    # --- 1. Import the desired experiment definitions ---
    # This is the only application-specific part of this file.
    from engine.environments.democracy.experiments import define_democracy_experiments
    experiments_to_run = define_democracy_experiments()
    
    # You could easily add more here, e.g.:
    # from environments.epidemiology.experiments import define_epi_experiments
    # experiments_to_run.extend(define_epi_experiments())
    
    print(f"Loaded {len(experiments_to_run)} experiment suites.")

    # --- 2. Generate all individual simulation jobs ---
    all_simulation_jobs = generate_run_list_from_experiments(experiments_to_run)
    
    if not all_simulation_jobs:
        print("No simulation jobs generated. Exiting.")
        exit()
        
    print(f"Generated a total of {len(all_simulation_jobs)} simulation runs to execute.")

    # --- 3. Set up output directory and runner ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"SimulationSuite_{timestamp}"
    output_dir = os.path.join("experiment_outputs", suite_name)
    os.makedirs(output_dir, exist_ok=True)

    runner = EnhancedParallelExperimentRunner(
        output_dir=output_dir,
        suite_timestamp=timestamp
    )

    # --- 4. Execute the experiment grid ---
    results_aggregator = runner.run_experiment_grid(all_simulation_jobs)
    
    # --- 5. Save results ---
    results_aggregator.save_results(os.path.join(output_dir, "final_aggregated_results"), timestamp)

    # --- 6. Run analysis ---
    print("\n--- Starting Post-Hoc Analysis ---")
    data_df = results_aggregator.get_concatenated_data()
    metadata_df = results_aggregator.get_metadata_summary()
    if not data_df.empty:
        # Assuming you have a general analysis script
        # run_full_analysis(data_df, metadata_df, output_dir)
        print(f"Analysis complete. Results are in: {output_dir}")
    else:
        print("No data was generated, skipping analysis.")
        
    print(f"\n✅ Entire suite finished in {(time.time() - start_time) / 60:.2f} minutes.")