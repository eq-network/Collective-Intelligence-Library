# execution/instrumentation/metrics.py
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Callable
import matplotlib.pyplot as plt

from core.graph import GraphState
from core.category import Transform

class MetricsCollector:
    """
    Collects and computes performance metrics during simulation.
    Focuses on key democratic decision-making indicators.
    """
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize metrics collector with optional metrics list.
        
        Args:
            metrics: List of metrics to collect (default: basic set)
        """
        self.metrics = metrics or ["time", "resources", "decision_quality"]
        self.start_time = None
        self._metrics = {}
        self.metrics_history = []
    
    def start_collection(self) -> None:
        """Start a metrics collection period."""
        self.start_time = time.time()
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record a set of metrics."""
        self._metrics.update(metrics)
    
    def finish_collection(self, transform: Transform, 
                          state_before: GraphState, 
                          state_after: GraphState) -> None:
        """
        Finish collection and compute metrics.
        
        Args:
            transform: The transformation that was applied
            state_before: State before transformation
            state_after: State after transformation
        """
        # Time metrics
        if "time" in self.metrics and self.start_time is not None:
            self._metrics["execution_time"] = time.time() - self.start_time
        
        # Resource metrics
        if "resources" in self.metrics:
            resources_before = state_before.global_attrs.get("total_resources", 0)
            resources_after = state_after.global_attrs.get("total_resources", 0)
            
            self._metrics["resources_before"] = resources_before
            self._metrics["resources_after"] = resources_after
            self._metrics["resource_change"] = resources_after - resources_before
            self._metrics["resource_change_pct"] = ((resources_after / resources_before) - 1) * 100 if resources_before > 0 else 0
        
        # Decision quality metrics (comparing to prediction market)
        if "decision_quality" in self.metrics:
            if ("current_decision" in state_after.global_attrs and 
                "prediction_market" in state_after.global_attrs):
                
                decision = state_after.global_attrs["current_decision"]
                prediction = state_after.global_attrs["prediction_market"]
                
                # How well aligned the decision was with prediction market signals
                if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                    best_option = np.argmax(prediction)
                    self._metrics["decision_optimality"] = 1.0 if decision == best_option else 0.0
                    self._metrics["decision_vs_optimum"] = prediction[decision] / prediction[best_option] if prediction[best_option] > 0 else 0.0
        
        # Adversarial impact metrics
        if "adversarial_impact" in self.metrics and "is_adversarial" in state_after.node_attrs:
            # Calculate weighted influence of adversarial agents
            if "voting_power" in state_after.node_attrs:
                adversarial = state_after.node_attrs["is_adversarial"]
                voting_power = state_after.node_attrs["voting_power"]
                
                total_power = np.sum(voting_power)
                adversarial_power = np.sum(voting_power * adversarial)
                
                self._metrics["adversarial_influence"] = adversarial_power / total_power if total_power > 0 else 0.0
        
        # Add round information
        self._metrics["round"] = state_after.global_attrs.get("round", 0)
        
        # Copy the metrics to avoid reference issues
        self.metrics_history.append(dict(self._metrics))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        return dict(self._metrics)
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the full metrics history."""
        return self.metrics_history
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to a pandas DataFrame."""
        return pd.DataFrame(self.metrics_history)
    
    def plot_resource_changes(self, figsize=(10, 6)):
        """Plot resource changes over time."""
        df = self.get_dataframe()
        if df.empty or 'resource_change_pct' not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(df['round'], df['resource_change_pct'])
        ax.set_title('Resource Change by Round (%)')
        ax.set_xlabel('Round')
        ax.set_ylabel('Change (%)')
        ax.grid(True, alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        return fig
    
    def plot_decision_quality(self, figsize=(10, 6)):
        """Plot decision quality metrics over time."""
        df = self.get_dataframe()
        if df.empty or 'decision_vs_optimum' not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df['round'], df['decision_vs_optimum'], marker='o', linestyle='-')
        ax.set_title('Decision Quality Over Time')
        ax.set_xlabel('Round')
        ax.set_ylabel('Decision Quality (vs Optimum)')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_mechanism_comparison(self, mechanism_metrics: Dict[str, pd.DataFrame], figsize=(12, 8)):
        """
        Plot comparison between different democratic mechanisms.
        
        Args:
            mechanism_metrics: Dictionary mapping mechanism names to DataFrames with metrics
            
        Returns:
            Matplotlib figure
        """
        if not mechanism_metrics:
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot total resources over time
        for name, df in mechanism_metrics.items():
            if 'resources_after' in df.columns and 'round' in df.columns:
                axes[0].plot(df['round'], df['resources_after'], marker='o', label=name)
        
        axes[0].set_title('Resource Comparison Across Mechanisms')
        axes[0].set_ylabel('Total Resources')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot decision quality over time
        for name, df in mechanism_metrics.items():
            if 'decision_vs_optimum' in df.columns and 'round' in df.columns:
                axes[1].plot(df['round'], df['decision_vs_optimum'], marker='o', label=name)
        
        axes[1].set_title('Decision Quality Comparison')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Decision Quality')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig