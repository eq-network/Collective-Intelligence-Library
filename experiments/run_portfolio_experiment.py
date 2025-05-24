import time
import os
from datetime import datetime
import multiprocessing # For freeze_support and cpu_count
from typing import List, Dict, Any # For type hinting

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Local imports from the 'experiments' package
from .experiment_config import ExperimentDefinition, generate_all_run_parameters, SingleRunParameters
from .runner import ParallelExperimentRunner
from .analysis import AnalysisPipeline
from .results import ResultsAggregator # Runner returns this directly

# Import the config factory functions to ensure they are known (though worker calls them by name via dict)
from environments.democracy.configuration import create_thesis_baseline_config, create_thesis_highvariance_config




def define_all_experiments() -> List[ExperimentDefinition]:
    """Defines all experimental setups to be run."""
    experiments = []

    # Define the Baseline Experiment Setup
    #baseline_exp = ExperimentDefinition(
    #    name="Baseline_3Crop_StdVar",
    #    config_factory_func_name="create_thesis_baseline_config", # Worker will use this name
    #    mechanisms_to_test=["PDD", "PRD", "PLD"],
    #    adversarial_proportions_to_sweep=[0.1, 0.25, 0.4],
    #    num_replications_per_setting=10, # Example: Increased replications
    #    base_seed_for_experiment=20240801,
    #    llm_model='google/gemini-2.5-flash-preview-05-20' # Or your preferred model
    #)
    #experiments.append(baseline_exp)

    # Define the High Variance Experiment Setup
    high_variance_exp = ExperimentDefinition(
        name="HighVariance_5Crop_MorePortfolios",
        config_factory_func_name="create_thesis_highvariance_config", # Worker uses this name
        mechanisms_to_test=["PDD", "PRD", "PLD"],
        adversarial_proportions_to_sweep=[0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8],
        num_replications_per_setting=2, # Example: Increased replications
        base_seed_for_experiment=20240802,
        llm_model='google/gemini-2.5-flash-preview-05-20' # Or your preferred model
    )
    experiments.append(high_variance_exp)
    
    return experiments

def main():
    start_overall_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    suite_name = f"PortfolioDemocracySuite_{timestamp}" # Give the overall run a name
    print(f"===== MYCORRHIZA EXPERIMENT SUITE: {suite_name} =====")

    # Create a directory for this suite's results early
    suite_results_dir = f"experiment_outputs/{suite_name}"
    os.makedirs(suite_results_dir, exist_ok=True)


    # 1. Define all experiment configurations
    all_experiment_definitions = define_all_experiments()
    if not all_experiment_definitions:
        print("No experiments defined. Exiting.")
        return

    # 2. Generate all individual run parameter sets for the entire suite
    # The generate_all_run_parameters now takes a list of ExperimentDefinition
    flat_list_of_all_runs: List[Dict[str, Any]] = [] # Runner expects list of dicts
    current_run_id_offset = 0
    for exp_def in all_experiment_definitions:
        print(f"\nGenerating runs for Experiment: {exp_def.name}")
        # Convert SingleRunParameters objects to dicts for ProcessPoolExecutor
        single_run_param_objects = generate_all_run_parameters([exp_def], global_run_id_offset=current_run_id_offset)
        for srp_obj in single_run_param_objects:
            flat_list_of_all_runs.append(srp_obj.__dict__) # Convert dataclass to dict
        
        print(f"  Generated {len(single_run_param_objects)} runs for this experiment definition.")
        current_run_id_offset += len(single_run_param_objects)
        
    if not flat_list_of_all_runs:
        print("No runs generated from experiment definitions. Exiting.")
        return
    print(f"\nTotal simulation runs to execute across all experiments: {len(flat_list_of_all_runs)}")

    # 3. Initialize and run experiments in parallel
    # Leave one core free, or limit to a reasonable number like 8 if many cores
    max_workers = min(multiprocessing.cpu_count() - 1, 8) if multiprocessing.cpu_count() > 1 else 1
    experiment_runner = ParallelExperimentRunner(
        output_dir=suite_results_dir, # Pass output directory
        suite_timestamp=timestamp,    # Pass timestamp
        max_workers=max_workers
    ) 
    
    print(f"\nStarting experiment execution with up to {max_workers} workers...")
    results_aggregator: ResultsAggregator = experiment_runner.run_experiment_grid(flat_list_of_all_runs)
    
    print("\nExperiment execution finished.")

    # 4. Save final aggregated results (will overwrite the last intermediate save)
    print("\nSaving final aggregated results...")
    results_aggregator.save_results(os.path.join(suite_results_dir, "aggregated_final"), timestamp)

    # 5. Perform Analysis
    print("\nStarting analysis...")
    data_df = results_aggregator.get_concatenated_data()
    metadata_df = results_aggregator.get_metadata_summary()

    if not data_df.empty and not metadata_df.empty:
        analysis_pipeline = AnalysisPipeline(data_df, metadata_df, output_dir=suite_results_dir)
        analysis_pipeline.run_default_analysis(timestamp) # This will now iterate through 'experiment_name'
        print(f"Analysis complete. Outputs in: {suite_results_dir}")
    else:
        print("No data aggregated to analyze.")

    end_overall_time = time.time()
    total_duration_seconds = end_overall_time - start_overall_time
    hours, rem = divmod(total_duration_seconds, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nTotal experiment suite duration: {int(hours):02d}h {int(mins):02d}m {secs:.2f}s.")
    print(f"=================== SUITE FINISHED: {suite_name} ===================")

if __name__ == "__main__":
    # Good practice for multiprocessing, especially on Windows,
    # to ensure child processes don't re-execute module-level code.
    multiprocessing.freeze_support() 
    main()