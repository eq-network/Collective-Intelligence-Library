import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compare_resource_progression(csv_path, save_path=None):
    """
    Creates a direct visualization comparing resource progression across simulation runs
    identified by specific row indices in the CSV file.
    
    Args:
        csv_path: Path to the CSV file containing simulation results
        save_path: Optional path to save the visualization (if None, won't save)
        
    Returns:
        Matplotlib figure object
    """
    # Step 1: Data acquisition and validation
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Step 2: Validate required column exists
    if 'resources_after' not in df.columns:
        raise ValueError("CSV file must contain 'resources_after' column")
    
    # Step 3: Partition data into runs based on specified row indices
    # Extracting the two runs based on row positions
    run1_start_idx = 1  # Zero-indexed row 1 (second row)
    run2_start_idx = 31  # Zero-indexed row 31 (32nd row)
    
    # Ensure we have enough rows for both runs
    if len(df) <= run2_start_idx:
        raise ValueError(f"CSV file has insufficient rows. Has {len(df)}, needs at least {run2_start_idx+1}")
    
    # Extract run data and assign round numbers
    run1_data = df.iloc[run1_start_idx:run2_start_idx].copy()
    run2_data = df.iloc[run2_start_idx:].copy()
    
    # Add explicit round numbers starting from 0
    run1_data['round'] = range(len(run1_data))
    run2_data['round'] = range(len(run2_data))
    
    # Step 4: Create visualization with precise color assignments
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use consistent colors from a standard colormap
    colors = ['#1f77b4', '#ff7f0e']  # Standard matplotlib blue and orange
    
    # Plot each run with appropriate labeling
    ax.plot(
        run1_data['round'], 
        run1_data['resources_after'],
        marker='o',
        label=f"Run 1 (Adv={run1_data['adv_prop_total'].iloc[0]})",
        color=colors[0],
        linewidth=2
    )
    
    ax.plot(
        run2_data['round'], 
        run2_data['resources_after'],
        marker='o',
        label=f"Run 2 (Adv={run2_data['adv_prop_total'].iloc[0]})",
        color=colors[1],
        linewidth=2
    )
    
    # Step 5: Configure visualization parameters for optimal readability
    ax.set_xlabel('Round')
    ax.set_ylabel('Resources')
    ax.set_title('Resource Progression Comparison Between Runs')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Step 6: Handle output persistence
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    return fig

# Example usage
if __name__ == "__main__":
    fig = compare_resource_progression("simulation_results_llm_raw.csv", "resource_comparison.png")
    plt.show()