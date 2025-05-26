import pandas as pd
import os
from datetime import datetime
import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
# Adjust this path if reanalyze_experiment.py is not in the same directory as run_portfolio_experiment.py
script_dir = Path(__file__).resolve().parent 
root_dir = script_dir.parent # Assuming experiments folder is one level up from Mycorrhiza root
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from experiments.analysis import TimelineAnalysisPipeline # Use TimelineAnalysisPipeline directly

def reanalyze_data(timeline_csv_path: str, metadata_csv_path: str, output_base_dir: str):
    """
    Loads existing timeline and metadata CSVs and runs the comprehensive timeline analysis.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a unique output directory for this re-analysis run
    analysis_name = Path(timeline_csv_path).stem.replace("aggregated_final_timeline_data_", "")
    output_dir = os.path.join(output_base_dir, f"Reanalysis_{analysis_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Re-analysis output directory: {output_dir}")

    try:
        timeline_df = pd.read_csv(timeline_csv_path)
        print(f"Successfully loaded timeline data from: {timeline_csv_path}")
    except FileNotFoundError:
        print(f"Error: Timeline data CSV not found at {timeline_csv_path}")
        return
    except Exception as e:
        print(f"Error loading timeline data CSV: {e}")
        return

    try:
        metadata_df = pd.read_csv(metadata_csv_path)
        print(f"Successfully loaded metadata from: {metadata_csv_path}")
    except FileNotFoundError:
        print(f"Error: Metadata CSV not found at {metadata_csv_path}. Some analyses might be limited.")
        # Create a minimal metadata_df if not found, to allow some plots to run
        if not timeline_df.empty:
            print("Creating minimal metadata from timeline data for basic analysis.")
            # Attempt to create a basic metadata_df if timeline_df is loaded
            # This is a fallback and might not have all necessary info for all plots
            unique_runs = timeline_df[['run_id', 'mechanism', 'adversarial_proportion_total']].drop_duplicates()
            metadata_df = unique_runs.copy()
            metadata_df['status'] = 'success' # Assume success for re-analysis purposes
            metadata_df['rounds_completed'] = timeline_df.groupby('run_id')['round'].max().values
        else:
            metadata_df = pd.DataFrame() # Keep it empty if timeline also failed
            
    except Exception as e:
        print(f"Error loading metadata CSV: {e}")
        metadata_df = pd.DataFrame()


    if not timeline_df.empty:
        # Ensure 'chosen_portfolio_idx' exists, add if not (with default -1)
        # This is because your provided CSV doesn't have it, but analysis.py expects it.
        if 'chosen_portfolio_idx' not in timeline_df.columns:
            print("Warning: 'chosen_portfolio_idx' not found in timeline_df. Adding with default -1.")
            timeline_df['chosen_portfolio_idx'] = -1
        
        # Ensure 'replication_run_index' exists, add if not (with default 0)
        if 'replication_run_index' not in timeline_df.columns:
            print("Warning: 'replication_run_index' not found in timeline_df. Adding with default 0.")
            timeline_df['replication_run_index'] = 0


        # If metadata_df is still empty or lacks critical columns, try to derive more from timeline_df
        if metadata_df.empty or not {'run_id', 'mechanism', 'adversarial_proportion_total'}.issubset(metadata_df.columns):
            if not timeline_df.empty:
                print("Re-creating minimal metadata from timeline data as metadata was incomplete.")
                unique_runs_info = timeline_df[['run_id', 'mechanism', 'adversarial_proportion_total']].drop_duplicates()
                metadata_df = unique_runs_info.copy()
                metadata_df['status'] = 'success' # Assume success
                # Get rounds_completed from timeline_df
                rounds_completed_map = timeline_df.groupby('run_id')['round'].max()
                metadata_df['rounds_completed'] = metadata_df['run_id'].map(rounds_completed_map)
            else:
                print("Cannot proceed: Timeline data is empty and metadata is insufficient.")
                return


        print("\nStarting timeline re-analysis...")
        # Use TimelineAnalysisPipeline directly
        timeline_analyzer = TimelineAnalysisPipeline(timeline_df, metadata_df, output_dir=output_dir)
        timeline_analyzer.run_comprehensive_timeline_analysis(timestamp)
        print(f"Timeline re-analysis complete. Outputs in: {output_dir}")
    else:
        print("No timeline data loaded to analyze.")

if __name__ == "__main__":
    # --- CONFIGURE THESE PATHS ---
    # Path to your specific timeline data CSV
    timeline_csv_file = r"experiment_outputs\TimelinePortfolioDemocracySuite_20250525_224038\aggregated_final_timeline_data_20250525_224038.csv\aggregated_final_timeline_data_20250525_224038.csv"
    
    # Path to the corresponding metadata CSV
    # Construct it based on the timeline path or specify directly
    metadata_csv_file = r"experiment_outputs\TimelinePortfolioDemocracySuite_20250525_224038\aggregated_final_metadata_20250525_224038.csv"
    # For your specific file, the metadata might be one level up and named differently
    timeline_dir = Path(timeline_csv_file).parent.parent # Go up two levels from the CSV itself
    timestamp_from_path = timeline_dir.name.split('_')[-2] + "_" + timeline_dir.name.split('_')[-1] # Extracts YYYYMMDD_HHMMSS
    
    # Try to infer metadata path, adjust if your naming convention is different
    metadata_csv_file = timeline_dir / f"aggregated_final_metadata_{timestamp_from_path}.csv"
    
    # Base directory where re-analysis outputs will be saved
    reanalysis_output_base = "experiment_outputs_reanalyzed"
    # --- END CONFIGURATION ---

    print(f"Attempting to re-analyze:")
    print(f"  Timeline: {timeline_csv_file}")
    print(f"  Metadata: {metadata_csv_file}")

    reanalyze_data(timeline_csv_file, str(metadata_csv_file), reanalysis_output_base)
