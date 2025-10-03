# run_portfolio_experiment.py
"""
Defines and executes the specific experimental suite for the
Portfolio Democracy simulations using the new generic runner framework.
"""
import time
import os
import multiprocessing
import itertools
from datetime import datetime
from typing import List, Dict, Any

from .runner import EnhancedParallelExperimentRunner
from .analysis import AnalysisPipeline # Assuming this is adapted for new data format

def define_experiments() -> List[Dict[str, Any]]:
    """
    Defines the experimental configurations to be run.
    Each dictionary represents a complete experimental setup.
    """
    experiments = []

    # --- Experiment 1: Baseline Comparison with Hardcoded Adversaries ---
    # This is a fast, deterministic experiment perfect for testing and baselining.
    baseline_exp = {
        "name": "Stable_Baseline_Hardcoded_Adversary",
        "num_replications_per_setting": 5,
        "base_seed": 202401,
        "parameter_sweeps": {
            "mechanism": ["PDD", "PLD"],
            "adversarial_proportion_total": [0.0, 0.2, 0.4, 0.6],
        },
        # Use hardcoded agents for both roles for this baseline
        "static_params": {
            "aligned_agent_path": "environments.democracy.agents.hardcoded.HardcodedAlignedAgent",
            "adversarial_agent_path": "environments.democracy.agents.hardcoded.HardcodedAdversarialAgent",
            "llm_model": None, # No LLM needed
        }
    }
    experiments.append(baseline_exp)

    # --- Experiment 2: LLM Aligned vs. Hardcoded Adversaries ---
    # This tests the LLM agents in a controlled environment.
    llm_aligned_exp = {
        "name": "Stable_LLM_Aligned_vs_Hardcoded_Adv",
        "num_replications_per_setting": 3, # Fewer reps since LLM is slower
        "base_seed": 202402,
        "parameter_sweeps": {
            "mechanism": ["PDD", "PLD"],
            "adversarial_proportion_total": [0.3, 0.5],
        },
        "static_params": {
            "aligned_agent_path": "environments.democracy.agents.llm_agents.AlignedHeuristicAgent",
            "adversarial_agent_path": "environments.democracy.agents.hardcoded.HardcodedAdversarialAgent",
            "llm_model": "openai/gpt-4o-mini",
        }
    }
    experiments.append(llm_aligned_exp)

    return experiments

def generate_all_run_params(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expands a list of experiment definitions into a flat list of single-run parameters.
    """
    all_runs = []
    run_counter = 0

    for exp in experiments:
        sweep_keys = list(exp["parameter_sweeps"].keys())
        sweep_values = list(exp["parameter_sweeps"].values())
        
        for combo in itertools.product(*sweep_values):
            run_base_params = exp["static_params"].copy()
            for i, key in enumerate(sweep_keys):
                run_base_params[key] = combo[i]
            
            for rep in range(exp["num_replications_per_setting"]):
                run_params = run_base_params.copy()
                run_params['run_id'] = run_counter
                run_params['experiment_name'] = exp['name']
                run_params['replication_index'] = rep
                run_params['unique_config_seed'] = exp['base_seed'] + run_counter
                
                all_runs.append(run_params)
                run_counter += 1
                
    return all_runs

def main():
    """Main execution function."""
    start_overall_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"StableDemocracySuite_{timestamp}"
    
    print(f"===== STARTING EXPERIMENT SUITE: {suite_name} =====")
    
    # Create results directory
    suite_results_dir = os.path.join("experiment_outputs", suite_name)
    os.makedirs(suite_results_dir, exist_ok=True)

    # 1. Define experiments and generate all individual run configurations
    experiments_to_run = define_experiments()
    all_run_parameters = generate_all_run_params(experiments_to_run)
    print(f"Generated a total of {len(all_run_parameters)} simulation runs.")

    # 2. Initialize and start the parallel runner
    runner = EnhancedParallelExperimentRunner(
        output_dir=suite_results_dir,
        suite_timestamp=timestamp,
        max_workers=min(os.cpu_count() - 1, 8)
    )
    results_aggregator = runner.run_experiment_grid(all_run_parameters)

    # 3. Save final aggregated results
    results_aggregator.save_results(os.path.join(suite_results_dir, "final_aggregated"), timestamp)

    # 4. Run analysis on the results
    print("\n--- Starting Analysis ---")
    data_df = results_aggregator.get_concatenated_data()
    metadata_df = results_aggregator.get_metadata_summary()
    if not data_df.empty:
        analysis_pipeline = AnalysisPipeline(data_df, metadata_df, output_dir=suite_results_dir)
        analysis_pipeline.run_comprehensive_timeline_analysis(timestamp)
        print(f"Analysis complete. Visualizations saved in: {suite_results_dir}")
    else:
        print("No data available for analysis.")

    end_overall_time = time.time()
    print(f"\nSuite finished in {(end_overall_time - start_overall_time):.2f} seconds.")

if __name__ == "__main__":
    # This is important for compatibility on some systems (like Windows)
    multiprocessing.freeze_support()
    main()