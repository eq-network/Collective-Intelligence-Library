import pandas as pd
import os
from datetime import datetime
import sys
from pathlib import Path

# --- Adjust these paths as necessary ---
# Assuming heatmaps.py is in Mycorrhiza/services/helper/
# And this script is in Mycorrhiza/ or Mycorrhiza/experiments/

# Add project root to sys.path to allow imports from services
# This assumes this script is in the Mycorrhiza root or one level down (e.g., experiments)
try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent # If script is in root
    if project_root.name == "experiments" or project_root.name == "services": # If script is one level down
        project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError: # If running in an interactive environment like Jupyter
    # Manually set project_root if __file__ is not defined
    # You might need to adjust this path
    project_root = Path("C:/Users/Jonas/Documents/GitHub/Mycorrhiza").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


from services.helper.heatmaps import HeatmapVisualizer

def main_analysis(csv_file_path: str, output_directory: str):
    """
    Loads data from a CSV and runs the heatmap visualizations.
    """
    print(f"Loading data from: {csv_file_path}")
    try:
        timeline_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if timeline_df.empty:
        print("The loaded CSV is empty. No analysis will be performed.")
        return

    # Create a timestamp for the output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure the output directory for this specific analysis exists
    # You might want to create a subdirectory for each CSV analysis
    analysis_output_dir = os.path.join(output_directory, f"heatmap_analysis_{Path(csv_file_path).stem}_{timestamp}")
    os.makedirs(analysis_output_dir, exist_ok=True)
    print(f"Output will be saved to: {analysis_output_dir}")

    # Instantiate the visualizer
    visualizer = HeatmapVisualizer(timeline_df=timeline_df, output_dir=analysis_output_dir)

    # Call the desired plotting functions
    print("\nGenerating enhanced trajectories plot...")
    visualizer.plot_enhanced_trajectories(timestamp=timestamp)

    print("\nGenerating trajectory heatmaps...")
    visualizer.plot_trajectory_heatmaps(timestamp=timestamp)

    print("\nAnalyzing phase transitions...")
    visualizer.analyze_phase_transitions(timestamp=timestamp)

    print("\nGenerating comparative resilience plot...")
    visualizer.plot_comparative_resilience(timestamp=timestamp)

    print("\nGenerating adversarial regime analysis plot...")
    visualizer.plot_adversarial_regime_analysis(timestamp=timestamp)

    print("\nGenerating regime performance summary plot...")
    visualizer.plot_regime_performance_summary(timestamp=timestamp)

    print("\nGenerating regime transition analysis...")
    visualizer.generate_regime_transition_analysis(timestamp=timestamp)

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    # Replace with the actual path to your CSV file
    csv_to_analyze = r"experiment_outputs\TimelinePortfolioDemocracySuite_20250526_122811- Gemini Flash 2.0\aggregated_final_timeline_data_20250526_122811.csv\aggregated_final_timeline_data_20250526_122811.csv"
    
    # Replace with your desired base output directory for these plots
    base_output_dir = r"experiment_outputs\TimelinePortfolioDemocracySuite_20250526_122811- Gemini Flash 2.0"
    # --- END USER CONFIGURATION ---

    if not os.path.exists(csv_to_analyze):
        print(f"ERROR: The specified CSV file does not exist: {csv_to_analyze}")
        print("Please update the 'csv_to_analyze' variable in the script.")
    else:
        main_analysis(csv_file_path=csv_to_analyze, output_directory=base_output_dir)
