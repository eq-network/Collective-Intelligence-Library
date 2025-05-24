# progress_tracker.py

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import statistics

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class SimulationProgressTracker:
    """
    Advanced progress tracker with ETA calculation and performance monitoring.
    
    Architectural Purpose:
    - Provides real-time progress visualization with accurate time estimates
    - Tracks performance metrics to identify bottlenecks and optimization opportunities
    - Maintains historical data for trend analysis and capacity planning
    - Implements adaptive ETA calculation based on actual completion patterns
    
    Key Features:
    - Weighted ETA calculation using recent performance data
    - Process-specific performance tracking
    - Failure rate monitoring and reporting
    - Adaptive time window for accurate predictions
    """
    
    def __init__(self, total_tasks: int, update_interval: int = 5):
        """
        Initialize progress tracker with adaptive ETA calculation.
        
        Args:
            total_tasks: Total number of tasks to complete
            update_interval: Seconds between progress updates
        """
        self.total_tasks = total_tasks
        self.update_interval = update_interval
        
        # Core tracking state
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Performance monitoring
        self.completion_times: List[float] = []
        self.completion_timestamps: List[float] = []
        self.process_performance: Dict[int, List[float]] = {}
        
        # ETA calculation parameters
        self.eta_window_size = 10  # Number of recent completions for ETA
        self.min_samples_for_eta = 3  # Minimum samples before showing ETA
        
        print(f"\n🚀 Starting parallel execution of {total_tasks} simulations")
        print(f"⏱️  Progress updates every {update_interval} seconds")
        print("=" * 70)
    
    def record_completion(self, task_duration: float, process_id: int, success: bool = True) -> None:
        """
        Record task completion with performance metrics.
        
        Args:
            task_duration: Time taken for task completion in seconds
            process_id: ID of the process that completed the task
            success: Whether the task completed successfully
        """
        current_time = time.time()
        
        if success:
            self.completed_tasks += 1
            self.completion_times.append(task_duration)
            self.completion_timestamps.append(current_time)
            
            # Track per-process performance
            if process_id not in self.process_performance:
                self.process_performance[process_id] = []
            self.process_performance[process_id].append(task_duration)
            
        else:
            self.failed_tasks += 1
        
        # Provide update if enough time has passed
        if (current_time - self.last_update_time) >= self.update_interval:
            self._display_progress()
            self.last_update_time = current_time
    
    def _display_progress(self) -> None:
        """Display comprehensive progress information with ETA."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate completion statistics
        total_processed = self.completed_tasks + self.failed_tasks
        progress_pct = (total_processed / self.total_tasks) * 100
        success_rate = (self.completed_tasks / total_processed * 100) if total_processed > 0 else 0
        
        # Calculate ETA using weighted recent performance
        eta_str = self._calculate_eta(current_time)
        
        # Performance metrics
        avg_task_time = statistics.mean(self.completion_times) if self.completion_times else 0
        recent_avg = self._get_recent_average_time()
        
        # Progress bar visualization
        bar_length = 40
        filled_length = int(bar_length * progress_pct / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Display comprehensive progress information
        print(f"\n📊 Progress Update - {datetime.now().strftime('%H:%M:%S')}")
        print(f"[{bar}] {progress_pct:.1f}%")
        print(f"✅ Completed: {self.completed_tasks:3d} | ❌ Failed: {self.failed_tasks:2d} | 🎯 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Elapsed: {self._format_duration(elapsed_time):12s} | {eta_str}")
        print(f"📈 Avg Task Time: {avg_task_time:.1f}s | Recent Avg: {recent_avg:.1f}s")
        print(f"🖥️  Active Processes: {len(self.process_performance)}")
        print("-" * 70)
    
    def _calculate_eta(self, current_time: float) -> str:
        """
        Calculate ETA using weighted average of recent completion times.
        
        Implements adaptive ETA calculation that emphasizes recent performance
        to account for system warmup, resource contention, and performance drift.
        
        Returns:
            Formatted ETA string with confidence indicator
        """
        remaining_tasks = self.total_tasks - (self.completed_tasks + self.failed_tasks)
        
        if remaining_tasks <= 0:
            return "🎉 ETA: Complete!"
        
        if len(self.completion_times) < self.min_samples_for_eta:
            return "⏳ ETA: Calculating..."
        
        # Use weighted average favoring recent completions
        recent_times = self.completion_times[-self.eta_window_size:]
        
        # Apply exponential weighting to recent times
        weights = [1.5 ** i for i in range(len(recent_times))]
        weighted_avg = sum(t * w for t, w in zip(recent_times, weights)) / sum(weights)
        
        # Add buffer for process coordination overhead
        coordination_buffer = 1.1  # 10% buffer for parallel execution overhead
        estimated_time_per_task = weighted_avg * coordination_buffer
        
        # Calculate ETA
        eta_seconds = remaining_tasks * estimated_time_per_task
        eta_time = current_time + eta_seconds
        eta_datetime = datetime.fromtimestamp(eta_time)
        
        # Confidence indicator based on sample size and consistency
        confidence = min(100, (len(recent_times) / self.eta_window_size) * 100)
        confidence_indicator = "🎯" if confidence > 70 else "📊" if confidence > 40 else "⏳"
        
        return f"{confidence_indicator} ETA: {eta_datetime.strftime('%H:%M:%S')} ({self._format_duration(eta_seconds)} remaining)"
    
    def _get_recent_average_time(self) -> float:
        """Get average of recent completion times for trend analysis."""
        if not self.completion_times:
            return 0.0
        
        recent_count = min(5, len(self.completion_times))
        recent_times = self.completion_times[-recent_count:]
        return statistics.mean(recent_times)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes, secs = divmod(seconds, 60)
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            minutes, secs = divmod(remainder, 60)
            return f"{hours:.0f}h {minutes:.0f}m {secs:.0f}s"
    
    def get_final_summary(self) -> Dict[str, any]:
        """
        Generate comprehensive final execution summary.
        
        Returns:
            Dictionary containing detailed execution statistics
        """
        total_time = time.time() - self.start_time
        total_processed = self.completed_tasks + self.failed_tasks
        
        # Performance analysis
        if self.completion_times:
            avg_time = statistics.mean(self.completion_times)
            median_time = statistics.median(self.completion_times)
            std_dev = statistics.stdev(self.completion_times) if len(self.completion_times) > 1 else 0
        else:
            avg_time = median_time = std_dev = 0
        
        # Process performance analysis
        process_stats = {}
        for pid, times in self.process_performance.items():
            if times:
                process_stats[pid] = {
                    'completions': len(times),
                    'avg_time': statistics.mean(times),
                    'total_time': sum(times)
                }
        
        return {
            'execution_time': total_time,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': (self.completed_tasks / total_processed * 100) if total_processed > 0 else 0,
            'avg_task_time': avg_time,
            'median_task_time': median_time,
            'std_dev_task_time': std_dev,
            'total_task_time': sum(self.completion_times),
            'process_statistics': process_stats,
            'efficiency': (sum(self.completion_times) / total_time * 100) if total_time > 0 else 0
        }
    
    def display_final_summary(self) -> None:
        """Display comprehensive final execution summary."""
        summary = self.get_final_summary()
        
        print(f"\n🎯 EXECUTION SUMMARY")
        print("=" * 70)
        print(f"✅ Successfully Completed: {summary['completed_tasks']:3d}")
        print(f"❌ Failed:                {summary['failed_tasks']:3d}")
        print(f"🎯 Success Rate:          {summary['success_rate']:5.1f}%")
        print(f"⏱️  Total Execution Time:  {self._format_duration(summary['execution_time'])}")
        print(f"📊 Average Task Time:     {summary['avg_task_time']:5.1f}s")
        print(f"📈 Median Task Time:      {summary['median_task_time']:5.1f}s")
        print(f"🔧 Parallel Efficiency:   {summary['efficiency']:5.1f}%")
        print(f"🖥️  Active Processes:      {len(summary['process_statistics'])}")
        
        # Show top performing processes
        if summary['process_statistics']:
            top_processes = sorted(
                summary['process_statistics'].items(), 
                key=lambda x: x[1]['completions'], 
                reverse=True
            )[:3]
            
            print(f"\n🏆 Top Performing Processes:")
            for pid, stats in top_processes:
                print(f"   PID {pid}: {stats['completions']} tasks, {stats['avg_time']:.1f}s avg")
        
        print("=" * 70)