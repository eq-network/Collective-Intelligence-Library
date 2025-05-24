import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List, Dict, Any
from typing import Optional
import os # Added for path joining
import pandas as pd

from .worker import run_simulation_task # The function to be parallelized
from .progress_tracker import SimulationProgressTracker # Your progress tracker
from .results import ResultsAggregator

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class ParallelExperimentRunner:
    def __init__(self, output_dir: str, suite_timestamp: str, max_workers: Optional[int] = None):
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()
        self.output_dir = output_dir
        self.suite_timestamp = suite_timestamp
        print(f"ParallelRunner initialized with max_workers = {self.max_workers}")

    def run_experiment_grid(self, run_params_list: List[Dict[str, Any]]) -> ResultsAggregator:
        total_tasks = len(run_params_list)
        progress_tracker = SimulationProgressTracker(total_tasks=total_tasks, update_interval=5) # Adjust interval
        results_aggregator = ResultsAggregator()
        
        # For periodic saving
        processed_tasks_since_last_save = 0
        # Save every N tasks, or adjust as needed.
        # Consider total_tasks to set a reasonable interval, e.g., total_tasks // 10 or a fixed number.
        save_interval_tasks = max(1, min(50, total_tasks // 10 if total_tasks >= 10 else 5)) 

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_run_params = {
                executor.submit(run_simulation_task, params): params for params in run_params_list
            }

            for future in as_completed(future_to_run_params):
                run_params = future_to_run_params[future]
                try:
                    df_result, metadata_result = future.result()
                    results_aggregator.add_result(df_result, metadata_result)
                    is_success = metadata_result.get('status') == 'success'
                    duration = metadata_result.get('simulation_duration_sec', 0)
                    pid = metadata_result.get('worker_pid', 0)
                    progress_tracker.record_completion(task_duration=duration, process_id=pid, success=is_success)

                    processed_tasks_since_last_save += 1
                    # Save periodically or if it's the last task
                    if processed_tasks_since_last_save >= save_interval_tasks or \
                       (progress_tracker.completed_tasks + progress_tracker.failed_tasks) == total_tasks:
                        print(f"\n[Runner] Saving intermediate results ({progress_tracker.completed_tasks + progress_tracker.failed_tasks}/{total_tasks} tasks processed)...")
                        results_aggregator.save_results(os.path.join(self.output_dir, "aggregated_intermediate"), self.suite_timestamp)
                        processed_tasks_since_last_save = 0

                except Exception as exc:
                    print(f"Run generated an exception: {run_params['run_id']} -> {exc}")
                    # Create error metadata if future itself failed catastrophically
                    # (run_simulation_task should ideally catch and return its own error metadata)
                    error_meta = {**run_params, 'status': 'executor_error', 'error_message': str(exc)}
                    results_aggregator.add_result(pd.DataFrame(), error_meta) # Add empty df
                    progress_tracker.record_completion(task_duration=0, process_id=0, success=False)
        
        progress_tracker.display_final_summary() # Display summary after all tasks
        return results_aggregator