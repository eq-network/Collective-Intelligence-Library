# experiments/run_portfolio_experiment.py - SIMPLIFIED WITH TIMELINE INTEGRATION
import time
import os
from datetime import datetime
import multiprocessing
from typing import List, Dict, Any

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Local imports with enhanced timeline capabilities
from experiments.experiment_config import ExperimentDefinition, generate_all_run_parameters
from experiments.runner import EnhancedParallelExperimentRunner  # Enhanced for timeline data
from experiments.analysis import AnalysisPipeline  # Enhanced with timeline visualizations
from experiments.results import EnhancedResultsAggregator  # Enhanced for timeline data volume

# Import config factory functions
from environments.democracy.configuration import create_thesis_baseline_config, create_thesis_highvariance_config


def define_all_experiments() -> List[ExperimentDefinition]:
    """Define all experimental setups to be run."""
    experiments = []

    # Define the High Variance Experiment Setup for timeline analysis
    high_variance_exp = ExperimentDefinition(
        name="HighVariance_5Crop_Timeline",
        config_factory_func_name="create_thesis_highvariance_config",
        mechanisms_to_test=["PDD", "PRD", "PLD"],
        adversarial_proportions_to_sweep=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        num_replications_per_setting=2,
        base_seed_for_experiment=20240802,
        llm_model='google/gemini-2.5-flash-preview-05-20'
    )
    experiments.append(high_variance_exp)
    
    return experiments


def main():
    """Main execution function with timeline capabilities."""
    start_overall_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    suite_name = f"TimelinePortfolioDemocracySuite_{timestamp}"
    print(f"===== TIMELINE EXPERIMENT SUITE: {suite_name} =====")

    # Create results directory
    suite_results_dir = f"experiment_outputs/{suite_name}"
    os.makedirs(suite_results_dir, exist_ok=True)

    # 1. Define experiments
    all_experiment_definitions = define_all_experiments()
    if not all_experiment_definitions:
        print("No experiments defined. Exiting.")
        return

    # 2. Generate all simulation parameters
    flat_list_of_all_runs = []
    current_run_id_offset = 0
    
    for exp_def in all_experiment_definitions:
        print(f"\nGenerating runs for Experiment: {exp_def.name}")
        single_run_param_objects = generate_all_run_parameters([exp_def], global_run_id_offset=current_run_id_offset)
        
        for srp_obj in single_run_param_objects:
            flat_list_of_all_runs.append(srp_obj.__dict__)
        
        print(f"  Generated {len(single_run_param_objects)} runs for this experiment")
        current_run_id_offset += len(single_run_param_objects)
    
    if not flat_list_of_all_runs:
        print("No runs generated. Exiting.")
        return
        
    print(f"\nTotal timeline simulations to execute: {len(flat_list_of_all_runs)}")

    # 3. Execute experiments in parallel using enhanced runner
    max_workers = min(multiprocessing.cpu_count() - 1, 8) if multiprocessing.cpu_count() > 1 else 1
    
    experiment_runner = EnhancedParallelExperimentRunner(
        output_dir=suite_results_dir,
        suite_timestamp=timestamp,
        max_workers=max_workers
    )
    
    print(f"\nStarting parallel execution with {max_workers} workers...")
    results_aggregator = experiment_runner.run_experiment_grid(flat_list_of_all_runs)
    
    print("\nParallel execution complete.")

    # 4. Save final results
    print("\nSaving final aggregated results...")
    results_aggregator.save_results(
        os.path.join(suite_results_dir, "aggregated_final"), 
        timestamp
    )

    # 5. Run timeline analysis
    print("\nStarting timeline analysis...")
    data_df = results_aggregator.get_concatenated_data()
    metadata_df = results_aggregator.get_metadata_summary()

    if not data_df.empty and not metadata_df.empty:
        analysis_pipeline = AnalysisPipeline(data_df, metadata_df, output_dir=suite_results_dir)
        analysis_pipeline.run_default_analysis(timestamp)
        print(f"Timeline analysis complete. Outputs in: {suite_results_dir}")
    else:
        print("No data to analyze.")

    # Final summary
    end_overall_time = time.time()
    total_duration_seconds = end_overall_time - start_overall_time
    hours, rem = divmod(total_duration_seconds, 3600)
    mins, secs = divmod(rem, 60)
    
    print(f"\nTotal experiment suite duration: {int(hours):02d}h {int(mins):02d}m {secs:.2f}s")
    print(f"Results directory: {suite_results_dir}")
    print(f"=================== SUITE FINISHED: {suite_name} ===================")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()