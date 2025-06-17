# execution/instrumentation/metrics.py
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Callable

from algebra.graph import GraphState
from algebra.category import Transform

class MetricsCollector:
    """
    Extracts and computes metrics from simulation states.
    Now focused on pure extraction rather than collection during simulation.
    """
    
    @staticmethod
    def extract_metrics_from_states(
        state_history: List[GraphState],
        config_metadata: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Static method that extracts metrics from a sequence of states.
        
        Args:
            state_history: List of GraphState objects from simulation
            config_metadata: Optional metadata about the simulation
            
        Returns:
            DataFrame with computed metrics
        """
        if not state_history:
            return pd.DataFrame()
        
        config_metadata = config_metadata or {}
        metrics_list = []
        
        for i in range(1, len(state_history)):
            prev_state = state_history[i-1]
            curr_state = state_history[i]
            
            # Basic metrics from state
            metrics = {
                "round": curr_state.global_attrs.get("round_num", i-1),
                "resources_after": curr_state.global_attrs.get("current_total_resources", 0),
                "resources_before": prev_state.global_attrs.get("current_total_resources", 0),
            }
            
            # Calculate resource change
            resources_before = metrics["resources_before"]
            resources_after = metrics["resources_after"]
            metrics["resource_change"] = resources_after - resources_before
            
            if resources_before > 0:
                metrics["resource_change_pct"] = ((resources_after / resources_before) - 1) * 100
            else:
                metrics["resource_change_pct"] = 0
                
            # Decision metrics
            metrics["decision_idx"] = curr_state.global_attrs.get("current_decision", -1)
            
            # Decision quality metrics (comparing to prediction market)
            if ("current_decision" in curr_state.global_attrs and 
                "prediction_market" in curr_state.global_attrs):
                
                decision = curr_state.global_attrs["current_decision"]
                prediction = curr_state.global_attrs["prediction_market"]
                
                # How well aligned the decision was with prediction market signals
                if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                    best_option = np.argmax(prediction)
                    metrics["decision_optimality"] = 1.0 if decision == best_option else 0.0
                    
                    if prediction[best_option] > 0:
                        metrics["decision_vs_optimum"] = prediction[decision] / prediction[best_option]
                    else:
                        metrics["decision_vs_optimum"] = 0.0
            
            # Adversarial impact metrics
            if "is_adversarial" in curr_state.node_attrs and "voting_power" in curr_state.node_attrs:
                adversarial = curr_state.node_attrs["is_adversarial"]
                voting_power = curr_state.node_attrs["voting_power"]
                
                total_power = np.sum(voting_power)
                adversarial_power = np.sum(voting_power * adversarial)
                
                if total_power > 0:
                    metrics["adversarial_influence"] = adversarial_power / total_power
                else:
                    metrics["adversarial_influence"] = 0.0
            
            # Add metadata
            metrics.update(config_metadata)
            metrics_list.append(metrics)
        
        # Create DataFrame
        return pd.DataFrame(metrics_list)
    
    @staticmethod
    def extract_final_state_metrics(
        state_history: List[GraphState],
        config_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract metrics from only the final state.
        
        Args:
            state_history: List of GraphState objects from simulation
            config_metadata: Optional metadata about the simulation
            
        Returns:
            Dictionary with final state metrics
        """
        if not state_history:
            return {}
        
        config_metadata = config_metadata or {}
        
        # Get initial and final states
        initial_state = state_history[0]
        final_state = state_history[-1]
        
        # Extract metrics
        metrics = {
            "final_round": final_state.global_attrs.get("round_num", len(state_history) - 1),
            "initial_resources": initial_state.global_attrs.get("current_total_resources", 0),
            "final_resources": final_state.global_attrs.get("current_total_resources", 0),
            "total_resource_change": final_state.global_attrs.get("current_total_resources", 0) - 
                                    initial_state.global_attrs.get("current_total_resources", 0),
        }
        
        # Add metadata
        metrics.update(config_metadata)
        
        return metrics