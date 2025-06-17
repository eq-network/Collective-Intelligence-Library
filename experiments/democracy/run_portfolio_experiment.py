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
from experiments.democracy.experiment_config import ExperimentDefinition, generate_all_run_parameters
from experiments.democracy.runner import EnhancedParallelExperimentRunner  # Enhanced for timeline data
from experiments.democracy.analysis import AnalysisPipeline  # Enhanced with timeline visualizations
from experiments.democracy.results import EnhancedResultsAggregator  # Enhanced for timeline data volume

# Import config factory functions
from environments.democracy.random.configuration import create_thesis_baseline_config, create_thesis_highvariance_config
from environments.democracy.stable.configuration import (
    create_stable_democracy_config
)

def define_all_experiments() -> List[ExperimentDefinition]:
    """Define all experimental setups to be run."""
    experiments = []

    # --- Experiment 1: Competitive Framing with OpenAI GPT-4 ---
    competitive_openai_exp = ExperimentDefinition(
        name="StableBaseline_Competitive_OpenAI",
        config_factory_func_name="create_stable_democracy_config",
        mechanisms_to_test=["PRD", "PLD", "PDD"],
        adversarial_proportions_to_sweep=[0.4, 0.5, 0.6, 0.7, 0.8],
        num_replications_per_setting=10,
        base_seed_for_experiment=20240905,
        llm_model='openai/gpt-4o-mini',  # OpenRouter format for OpenAI GPT-4
        adversarial_framing="competitive"
    )
    experiments.append(competitive_openai_exp)

    # --- Experiment 2: Competitive Framing with Google Gemini ---
    competitive_gemini_exp = ExperimentDefinition(
        name="StableBaseline_Competitive_Gemini",
        config_factory_func_name="create_stable_democracy_config",
        mechanisms_to_test=["PRD", "PLD", "PDD"],
        adversarial_proportions_to_sweep=[0.4, 0.5, 0.6, 0.7, 0.8],
        num_replications_per_setting=10,
        base_seed_for_experiment=20240906,
        llm_model='google/gemini-2.5-flash-preview-05-20',  # OpenRouter format for Gemini
        adversarial_framing="competitive"
    )
    experiments.append(competitive_gemini_exp)

    # --- Experiment 3: Competitive Framing with Anthropic Claude ---
    competitive_claude_exp = ExperimentDefinition(
        name="StableBaseline_Competitive_Claude",
        config_factory_func_name="create_stable_democracy_config",
        mechanisms_to_test=["PRD", "PLD", "PDD"],
        adversarial_proportions_to_sweep=[0.4, 0.5, 0.6, 0.7, 0.8],
        num_replications_per_setting=10,
        base_seed_for_experiment=20240907,
        llm_model='anthropic/claude-3.5-haiku',  # OpenRouter format for Claude
        adversarial_framing="competitive"
    )
    experiments.append(competitive_claude_exp)
    
    return experiments
    


def main():
    """Main execution function with sequential experiment execution."""
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

    # Setup for parallel processing within each experiment
    max_workers = min(multiprocessing.cpu_count() - 1, 8) if multiprocessing.cpu_count() > 1 else 1

    # Run experiments sequentially
    for experiment_index, exp_def in enumerate(all_experiment_definitions, 1):
        experiment_start_time = time.time()
        print(f"\n\n====== Starting Experiment {experiment_index}/{len(all_experiment_definitions)}: {exp_def.name} ======")
        
        # Generate parameters for this specific experiment
        print(f"Generating runs for: {exp_def.name}")
        single_run_param_objects = generate_all_run_parameters([exp_def], global_run_id_offset=0)
        flat_list_of_runs = [srp_obj.__dict__ for srp_obj in single_run_param_objects]
        print(f"Generated {len(flat_list_of_runs)} runs for this experiment")

        # Create experiment-specific directory
        experiment_dir = os.path.join(suite_results_dir, f"{exp_def.name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)

        # Initialize runner for this experiment
        experiment_runner = EnhancedParallelExperimentRunner(
            output_dir=experiment_dir,
            suite_timestamp=timestamp,
            max_workers=max_workers
        )
        
        # Run this experiment
        print(f"Starting execution with {max_workers} workers...")
        results_aggregator = experiment_runner.run_experiment_grid(flat_list_of_runs)
        
        # Save results for this experiment
        results_aggregator.save_results(
            os.path.join(experiment_dir, "aggregated_results"), 
            timestamp
        )

        # Run analysis for this experiment
        print(f"\nAnalyzing results for: {exp_def.name}")
        data_df = results_aggregator.get_concatenated_data()
        metadata_df = results_aggregator.get_metadata_summary()

        if not data_df.empty and not metadata_df.empty:
            analysis_pipeline = AnalysisPipeline(data_df, metadata_df, output_dir=experiment_dir)
            analysis_pipeline.run_default_analysis(timestamp)
            print(f"Analysis complete for {exp_def.name}. Outputs in: {experiment_dir}")
        else:
            print(f"No data to analyze for {exp_def.name}")

        # Report experiment completion time
        exp_duration = time.time() - experiment_start_time
        hours, rem = divmod(exp_duration, 3600)
        mins, secs = divmod(rem, 60)
        print(f"\nExperiment {exp_def.name} completed in: {int(hours):02d}h {int(mins):02d}m {secs:.2f}s")
        print("=" * 80)

    # Final summary
    end_overall_time = time.time()
    total_duration_seconds = end_overall_time - start_overall_time
    hours, rem = divmod(total_duration_seconds, 3600)
    mins, secs = divmod(rem, 60)
    
    print(f"\nTotal suite duration: {int(hours):02d}h {int(mins):02d}m {secs:.2f}s")
    print(f"Results directory: {suite_results_dir}")
    print(f"=================== SUITE FINISHED: {suite_name} ===================")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()