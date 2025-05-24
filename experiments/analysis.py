import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class AnalysisPipeline:
    def __init__(self, aggregated_data_df: pd.DataFrame, aggregated_metadata_df: pd.DataFrame, output_dir: str):
        self.data_df = aggregated_data_df
        self.metadata_df = aggregated_metadata_df
        self.output_dir = output_dir # Use the passed output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Ensure it exists


    def generate_summary_stats(self) -> pd.DataFrame:
        # Filter for successful runs only for performance metrics
        successful_runs_meta = self.metadata_df[self.metadata_df['status'] == 'success'].copy()
        if successful_runs_meta.empty:
            print("No successful runs to analyze.")
            return pd.DataFrame()

        # Ensure 'adv_prop_total' exists and use a consistent name.
        # It seems 'adversarial_proportion_total' is what's in run_params.
        adv_prop_col = 'adversarial_proportion_total' 
        
        # Get final resources from the main data_df for successful runs
        # Need to join metadata (for run_id) with data_df (for final resources)
        last_round_data = self.data_df.loc[self.data_df.groupby('run_id')['round'].idxmax()]
        
        # Merge with successful_runs_meta to ensure we only consider successful runs
        # and have access to parameters like 'mechanism' and adv_prop_col
        analysis_df = pd.merge(
            successful_runs_meta[['run_id', 'mechanism', adv_prop_col, 'prediction_market_sigma']], # Select relevant cols from meta
            last_round_data[['run_id', 'resources_after']], # Select relevant cols from data
            on='run_id'
        )
        if analysis_df.empty:
            print("No data for successful runs after merging.")
            return pd.DataFrame()

        summary = analysis_df.groupby(['mechanism', adv_prop_col, 'prediction_market_sigma'])['resources_after'].agg(
            ['mean', 'std', 'median', 'count']
        ).reset_index()
        summary.rename(columns={
            'mean': 'avg_final_resources',
            'std': 'std_final_resources',
            'median': 'median_final_resources',
            'count': 'num_successful_runs'
        }, inplace=True)
        return summary

    def plot_final_resources_vs_adversarial(self, summary_df: pd.DataFrame, fixed_pm_sigma: float, timestamp: str):
        if summary_df.empty:
            print("No summary data to plot.")
            return

        plt.figure(figsize=(12, 7))
        plot_data = summary_df[summary_df['prediction_market_sigma'] == fixed_pm_sigma]
        
        adv_prop_col = 'adversarial_proportion_total' # Use the consistent name

        for mech in plot_data['mechanism'].unique():
            mech_data = plot_data[plot_data['mechanism'] == mech]
            plt.errorbar(
                mech_data[adv_prop_col],
                mech_data['avg_final_resources'],
                yerr=mech_data['std_final_resources'],
                label=mech,
                marker='o', capsize=5
            )
        
        plt.title(f'Average Final Resources vs. Adversarial Proportion (PM Sigma={fixed_pm_sigma})')
        plt.xlabel('Adversarial Agent Proportion')
        plt.ylabel('Average Final Resources')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_filename = os.path.join(self.output_dir, f"final_resources_plot_{timestamp}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved: {plot_filename}")

    # Add more plotting functions (e.g., survival rate, execution time distribution)

    def run_default_analysis(self, timestamp: str):
        summary_stats = self.generate_summary_stats()
        if not summary_stats.empty:
            # Assuming you have a fixed PM sigma you're interested in for this plot
            # This should come from your experiment config or be a parameter
            fixed_pm_sigma_for_plot = 0.25 # Example value
            if 'prediction_market_sigma' in summary_stats.columns:
                 self.plot_final_resources_vs_adversarial(summary_stats, fixed_pm_sigma_for_plot, timestamp)
            else:
                print("Warning: 'prediction_market_sigma' not in summary_stats, cannot generate specific plot.")

            summary_filename = os.path.join(self.output_dir, f"summary_stats_{timestamp}.csv")
            summary_stats.to_csv(summary_filename, index=False)
            print(f"Summary stats saved: {summary_filename}")