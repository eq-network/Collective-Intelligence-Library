# experiments/runner.py - ENHANCED FOR TIMELINE DATA
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import time
import psutil
import gc
from typing import List, Dict, Any, Optional, Tuple
import os
import pandas as pd

from .democracy.worker import run_simulation_task
from .democracy.progress_tracker import SimulationProgressTracker
from .results import EnhancedResultsAggregator

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class EnhancedParallelExperimentRunner:
    """
    ARCHITECTURAL PURPOSE: Optimized parallel execution for timeline-generating simulations.
    
    DESIGN PRINCIPLES:
    1. Memory-conscious parallel processing for large timeline datasets
    2. Enhanced progress tracking with trajectory-level insights  
    3. Intelligent resource management and load balancing
    4. Comprehensive error handling and recovery mechanisms
    
    OPTIMIZATION STRATEGIES:
    - Dynamic memory monitoring and garbage collection
    - Intelligent batching based on system resources
    - Enhanced progress reporting with timeline-specific metrics
    - Graceful degradation under memory pressure
    
    PERFORMANCE CONSIDERATIONS:
    - Timeline data is ~50× larger than previous summary data
    - Memory usage patterns require active management
    - I/O operations need optimization for larger datasets
    - Progress reporting can leverage timeline granularity
    """
    
    def __init__(self, output_dir: str, suite_timestamp: str, 
                 max_workers: Optional[int] = None,
                 memory_limit_gb: float = 8.0,
                 save_interval_adaptive: bool = True):
        """
        Initialize enhanced parallel runner with timeline optimizations.
        
        ARCHITECTURAL PARAMETERS:
        - memory_limit_gb: Trigger optimizations when memory usage exceeds this threshold
        - save_interval_adaptive: Dynamically adjust save frequency based on memory usage
        
        RESOURCE MANAGEMENT STRATEGY:
        - Monitor system memory usage continuously
        - Adapt processing parameters based on available resources
        - Implement graceful degradation under resource pressure
        """
        self.max_workers = max_workers if max_workers else min(multiprocessing.cpu_count(), 8)
        self.output_dir = output_dir
        self.suite_timestamp = suite_timestamp
        self.memory_limit_gb = memory_limit_gb
        self.save_interval_adaptive = save_interval_adaptive
        
        # Performance monitoring state
        self.peak_memory_usage_gb = 0.0
        self.memory_pressure_events = 0
        self.total_timeline_points_processed = 0
        self.average_timeline_points_per_sim = 0
        
        # Dynamic processing parameters
        self.current_save_interval = 50  # Initial save interval
        self.min_save_interval = 10     # Minimum interval under memory pressure
        self.max_save_interval = 100    # Maximum interval under optimal conditions
        
        print(f"[ENHANCED_RUNNER_INIT] Workers: {self.max_workers}, Memory limit: {memory_limit_gb}GB")
        print(f"[ENHANCED_RUNNER_INIT] Adaptive save interval: {save_interval_adaptive}")

    def _get_system_memory_info(self) -> Dict[str, float]:
        """
        SYSTEM MONITORING: Comprehensive memory usage analysis.
        
        METRICS COLLECTED:
        - Current process memory usage
        - System-wide memory availability  
        - Memory pressure indicators
        - Performance optimization triggers
        """
        process = psutil.Process()
        system = psutil.virtual_memory()
        
        process_memory_gb = process.memory_info().rss / 1024**3
        system_available_gb = system.available / 1024**3
        system_percent = system.percent
        
        # Update peak usage tracking
        if process_memory_gb > self.peak_memory_usage_gb:
            self.peak_memory_usage_gb = process_memory_gb
        
        return {
            'process_memory_gb': process_memory_gb,
            'system_available_gb': system_available_gb,
            'system_memory_percent': system_percent,
            'memory_pressure': process_memory_gb > self.memory_limit_gb
        }

    def _adjust_save_interval(self, memory_info: Dict[str, float]) -> None:
        """
        ADAPTIVE OPTIMIZATION: Dynamically adjust save frequency based on memory pressure.
        
        ADAPTATION STRATEGY:
        - Under memory pressure: Increase save frequency to prevent accumulation
        - Under optimal conditions: Decrease save frequency for performance
        - Gradual adjustments to prevent thrashing
        """
        if not self.save_interval_adaptive:
            return
            
        if memory_info['memory_pressure']:
            # Increase save frequency under memory pressure
            self.current_save_interval = max(self.min_save_interval, 
                                           int(self.current_save_interval * 0.8))
            if self.current_save_interval <= self.min_save_interval:
                self.memory_pressure_events += 1
                print(f"[MEMORY_PRESSURE] Save interval reduced to {self.current_save_interval} "
                      f"(Event #{self.memory_pressure_events})")
        else:
            # Gradually increase save interval under optimal conditions
            self.current_save_interval = min(self.max_save_interval,
                                           int(self.current_save_interval * 1.1))

    def _trigger_memory_optimization(self) -> None:
        """
        MEMORY OPTIMIZATION: Aggressive memory management under pressure conditions.
        
        OPTIMIZATION TECHNIQUES:
        1. Force garbage collection
        2. Clear unnecessary caches
        3. Optimize data structures where possible
        4. Report memory recovery results
        """
        memory_before = psutil.Process().memory_info().rss / 1024**3
        
        # Force garbage collection
        gc.collect()
        
        # Clear matplotlib cache if available
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
        
        memory_after = psutil.Process().memory_info().rss / 1024**3
        memory_recovered = memory_before - memory_after
        
        print(f"[MEMORY_OPTIMIZATION] Recovered {memory_recovered:.2f}GB "
              f"({memory_before:.2f}GB → {memory_after:.2f}GB)")

    def run_experiment_grid(self, run_params_list: List[Dict[str, Any]]) -> EnhancedResultsAggregator:
        """
        ENHANCED EXECUTION: Timeline-optimized parallel experiment execution.
        """
        total_tasks = len(run_params_list)
        
        # Initialize enhanced components
        progress_tracker = SimulationProgressTracker(
            total_tasks=total_tasks, 
            update_interval=5
        )
        
        results_aggregator = EnhancedResultsAggregator(
            memory_limit_gb=self.memory_limit_gb,
            chunk_size=5000  # Smaller chunks for timeline data
        )
        
        # Performance tracking initialization
        processed_tasks_since_last_save = 0
        execution_start_time = time.time()
        
        print(f"[ENHANCED_EXECUTION_START] Processing {total_tasks} simulations with timeline tracking")
        print(f"[INITIAL_MEMORY] {self._get_system_memory_info()['process_memory_gb']:.2f}GB")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_run_params = {
                executor.submit(run_simulation_task, params): params 
                for params in run_params_list
            }
            
            # Process results as they complete
            for future in as_completed(future_to_run_params):
                run_params = future_to_run_params[future]
                
                try:
                    # Get results and validate structure (NOW WITH ANOMALIES)
                    df_result, metadata_result, anomaly_log_result = future.result()
                    
                    # Timeline-specific validation
                    timeline_points = len(df_result) if not df_result.empty else 0
                    self.total_timeline_points_processed += timeline_points
                    
                    # Update average timeline points tracking
                    completed_tasks = progress_tracker.completed_tasks + progress_tracker.failed_tasks + 1
                    self.average_timeline_points_per_sim = (
                        self.total_timeline_points_processed / completed_tasks
                    )
                    
                    # Add to results with memory monitoring
                    memory_info = self._get_system_memory_info()
                    results_aggregator.add_result(df_result, metadata_result)
                    
                    # NEW: Add anomalies after the existing results_aggregator.add_result call
                    results_aggregator.add_anomalies(anomaly_log_result, metadata_result)
                    
                    # Enhanced progress tracking with timeline insights
                    is_success = metadata_result.get('status') == 'success'
                    duration = metadata_result.get('simulation_duration_sec', 0)
                    pid = metadata_result.get('worker_pid', 0)
                    
                    progress_tracker.record_completion(
                        task_duration=duration, 
                        process_id=pid, 
                        success=is_success
                    )
                    
                    processed_tasks_since_last_save += 1
                    
                    # Memory pressure management
                    if memory_info['memory_pressure']:
                        self._trigger_memory_optimization()
                        self._adjust_save_interval(memory_info)
                    
                    # Adaptive saving based on memory conditions
                    should_save = (
                        processed_tasks_since_last_save >= self.current_save_interval or
                        memory_info['memory_pressure'] or
                        (progress_tracker.completed_tasks + progress_tracker.failed_tasks) == total_tasks
                    )
                    
                    if should_save:
                        # Intermediate save with performance monitoring
                        save_start_time = time.time()
                        completed_count = progress_tracker.completed_tasks + progress_tracker.failed_tasks
                        
                        print(f"\n[INTERMEDIATE_SAVE] Saving results ({completed_count}/{total_tasks} completed)")
                        print(f"[TIMELINE_STATS] Processed {self.total_timeline_points_processed} timeline points "
                            f"(avg: {self.average_timeline_points_per_sim:.1f} per simulation)")
                        
                        results_aggregator.save_results(
                            os.path.join(self.output_dir, "aggregated_intermediate"), 
                            self.suite_timestamp,
                            use_compression=True
                        )
                        
                        save_duration = time.time() - save_start_time
                        print(f"[SAVE_COMPLETE] Duration: {save_duration:.2f}s, "
                            f"Memory: {self._get_system_memory_info()['process_memory_gb']:.2f}GB")
                        
                        processed_tasks_since_last_save = 0
                        
                        # Update adaptive parameters
                        self._adjust_save_interval(self._get_system_memory_info())
                
                except Exception as exc:
                    # Enhanced error handling with timeline context
                    print(f"[EXECUTION_ERROR] Run {run_params['run_id']} failed: {exc}")
                    
                    # Create structured error metadata with timeline context
                    error_metadata = {
                        **run_params,
                        'status': 'executor_error',
                        'error_message': str(exc),
                        'worker_pid': 0,
                        'simulation_duration_sec': 0,
                        'final_resources': 0.0,
                        'rounds_completed': 0,
                        'timeline_points_generated': 0
                    }
                    
                    # Create minimal error timeline data
                    error_timeline = pd.DataFrame([{
                        'run_id': run_params.get('run_id', -1),
                        'round': 0,
                        'resources_after': 0.0,
                        'mechanism': run_params.get('mechanism', 'unknown'),
                        'error': str(exc)
                    }])
                    
                    # Create error anomaly entry
                    error_anomaly_entry = [{
                        "anomaly_type": "RUNNER_ERROR", 
                        "run_id": run_params.get('run_id', -1), 
                        "error_message": str(exc)
                    }]
                    
                    results_aggregator.add_result(error_timeline, error_metadata)
                    results_aggregator.add_anomalies(error_anomaly_entry, run_params)  # Pass run_params as metadata
                    progress_tracker.record_completion(task_duration=0, process_id=0, success=False)
        
        # Final execution summary with timeline-specific metrics
        execution_duration = time.time() - execution_start_time
        final_memory_info = self._get_system_memory_info()
        
        print(f"\n[ENHANCED_EXECUTION_COMPLETE]")
        print(f"  Total execution time: {execution_duration:.2f}s")
        print(f"  Timeline points processed: {self.total_timeline_points_processed}")
        print(f"  Average timeline points per simulation: {self.average_timeline_points_per_sim:.1f}")
        print(f"  Peak memory usage: {self.peak_memory_usage_gb:.2f}GB")
        print(f"  Memory pressure events: {self.memory_pressure_events}")
        print(f"  Final memory usage: {final_memory_info['process_memory_gb']:.2f}GB")
        
        # Display final progress summary with timeline insights
        progress_tracker.display_final_summary()
        
        # Print results aggregator performance summary
        aggregator_perf = results_aggregator.get_performance_summary()
        print(f"\n[AGGREGATOR_PERFORMANCE]")
        print(f"  Memory utilization: {aggregator_perf['memory_utilization_pct']:.1f}%")
        print(f"  Memory warnings: {aggregator_perf['memory_warnings_triggered']}")
        print(f"  Data volume: {aggregator_perf['total_timeline_points']} timeline points")
        
        return results_aggregator

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        PERFORMANCE ANALYSIS: Comprehensive execution performance summary.
        
        METRICS REPORTED:
        - Timeline-specific performance statistics
        - Memory usage patterns and optimization effectiveness
        - Parallel processing efficiency
        - Resource utilization analysis
        """
        return {
            'total_timeline_points_processed': self.total_timeline_points_processed,
            'average_timeline_points_per_simulation': self.average_timeline_points_per_sim,
            'peak_memory_usage_gb': self.peak_memory_usage_gb,
            'memory_pressure_events': self.memory_pressure_events,
            'memory_limit_gb': self.memory_limit_gb,
            'final_save_interval': self.current_save_interval,
            'adaptive_optimization_enabled': self.save_interval_adaptive,
            'max_workers': self.max_workers
        }


# BACKWARD COMPATIBILITY: Alias for existing code
ParallelExperimentRunner = EnhancedParallelExperimentRunner