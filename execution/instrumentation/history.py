# execution/instrumentation/history.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from core.graph import GraphState
from core.category import Transform

class HistoryTracker:
    """
    Tracks the history of state transitions during simulation.
    Captures key state changes for later analysis.
    """
    
    def __init__(self, max_history: int = 100):
        """Initialize with optional limit on history size."""
        self.history = []
        self.max_history = max_history
        
    def record(self, transform: Transform, state_before: GraphState, state_after: GraphState) -> None:
        """
        Record a state transition.
        
        Args:
            transform: The transformation that was applied
            state_before: State before transformation
            state_after: State after transformation
        """
        # Extract key metrics from states
        entry = {
            'round': state_after.global_attrs.get('round', len(self.history)),
            'transform_name': transform.__name__ if hasattr(transform, '__name__') else str(transform),
            'resources_before': state_before.global_attrs.get('total_resources', 0),
            'resources_after': state_after.global_attrs.get('total_resources', 0),
            'resource_change': state_after.global_attrs.get('total_resources', 0) - 
                              state_before.global_attrs.get('total_resources', 0),
        }
        
        # Add crop allocation if available
        if 'current_decision' in state_after.global_attrs:
            entry['decision'] = state_after.global_attrs['current_decision']
            
        # Add vote distribution if available
        if 'vote_distribution' in state_after.global_attrs:
            entry['vote_distribution'] = state_after.global_attrs['vote_distribution'].tolist()
        
        # Add delegation info for PLD
        if 'delegation' in state_after.adj_matrices:
            # Count delegations received by each agent
            delegations = np.sum(state_after.adj_matrices['delegation'], axis=0)
            entry['delegations'] = delegations.tolist()
            
            # Top delegate (agent receiving most delegations)
            if np.max(delegations) > 0:
                entry['top_delegate'] = int(np.argmax(delegations))
                entry['top_delegate_power'] = float(np.max(delegations))
        
        # Append to history, maintaining max size
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full history list."""
        return self.history
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return history as a pandas DataFrame for analysis."""
        return pd.DataFrame(self.history)
    
    def plot_resource_trajectory(self, figsize=(10, 6)):
        """Plot resources over time."""
        import matplotlib.pyplot as plt
        
        df = self.get_dataframe()
        if df.empty or 'resources_after' not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df['round'], df['resources_after'], marker='o', linestyle='-')
        ax.set_title('Resource Trajectory')
        ax.set_xlabel('Round')
        ax.set_ylabel('Total Resources')
        ax.grid(True, alpha=0.3)
        
        # Add threshold line if available
        if df.shape[0] > 0 and 'resource_min_threshold' in df.iloc[0]:
            threshold = df.iloc[0]['resource_min_threshold']
            ax.axhline(y=threshold, color='r', linestyle='--', 
                      label=f'Survival Threshold ({threshold})')
            ax.legend()
        
        return fig