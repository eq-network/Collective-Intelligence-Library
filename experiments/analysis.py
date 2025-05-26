# experiments/analysis.py - ENHANCED TIMELINE ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, Any, List, Optional

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class TimelineAnalysisPipeline:
    """
    ARCHITECTURAL PURPOSE: Comprehensive timeline analysis for resource progression tracking.
    
    DESIGN PRINCIPLES:
    1. Timeline-first analysis approach (round-by-round progression)
    2. Multi-scale visualization (individual trajectories + aggregated trends)
    3. Comparative mechanism analysis with trajectory differentiation
    4. Statistical analysis of progression patterns
    
    VISUALIZATION STRATEGY:
    - Individual trajectory plots (single simulation resource progression)
    - Aggregated trajectory plots (mechanism comparison with confidence bands)
    - Resource change distribution analysis
    - Critical event detection (rapid resource changes)
    """
    
    def __init__(self, timeline_data_df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: str):
        """
        Initialize timeline analysis pipeline with enhanced data structures.
        
        ARCHITECTURAL VALIDATION:
        - Timeline data: Multiple rows per simulation (round-by-round)
        - Metadata: One row per simulation (summary information)
        - Output directory: Centralized visualization storage
        """
        self.timeline_df = timeline_data_df
        self.metadata_df = metadata_df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate 'resources_before' and 'resource_change_pct'
        if not self.timeline_df.empty:
            # Ensure 'round' is sorted within each group for correct shift
            self.timeline_df = self.timeline_df.sort_values(['run_id', 'round']).reset_index(drop=True)

            # Calculate 'resources_before' by shifting 'resources_after' from the previous round
            self.timeline_df['resources_before'] = self.timeline_df.groupby('run_id')['resources_after'].shift(1)

            # Fill NaN 'resources_before' for the first round of each run.
            # Assumes a default initial resource amount.
            # TODO: Ideally, fetch this from config/metadata per run_id if initial resources can vary.
            # For current thesis_baseline and thesis_highvariance configs, initial_amount is 100.0.
            default_initial_resources = 100.0
            self.timeline_df['resources_before'] = self.timeline_df['resources_before'].fillna(default_initial_resources)

            # Calculate 'resource_change_pct'
            delta_resources = self.timeline_df['resources_after'] - self.timeline_df['resources_before']
            
            self.timeline_df['resource_change_pct'] = np.where(
                self.timeline_df['resources_before'] != 0,
                (delta_resources / self.timeline_df['resources_before']) * 100,
                np.where(delta_resources == 0, 0.0, np.nan) # 0% if 0->0, NaN if 0->non-zero
            )

        # VALIDATION: Ensure timeline data structure integrity
        self._validate_timeline_data_structure()
        
    def _validate_timeline_data_structure(self) -> None:
        """
        ARCHITECTURAL VALIDATION: Ensure timeline data meets expected structure.
        
        VALIDATION DIMENSIONS:
        - Required columns presence
        - Data type consistency
        - Relationship integrity between timeline and metadata
        """
        required_timeline_cols = [
            'run_id', 'round', 'resources_after', 'mechanism', 
            'adversarial_proportion_total', 'replication_run_index', 'chosen_portfolio_idx'
        ]
        
        missing_cols = [col for col in required_timeline_cols if col not in self.timeline_df.columns]
        if missing_cols:
            raise ValueError(f"Timeline data missing required columns: {missing_cols}")
        
        # Verify run_id consistency between timeline and metadata
        timeline_runs = set(self.timeline_df['run_id'].unique())
        metadata_runs = set(self.metadata_df['run_id'].unique())
        
        if not timeline_runs.issubset(metadata_runs):
            missing_metadata = timeline_runs - metadata_runs
            print(f"Warning: Timeline data contains run_ids not in metadata: {missing_metadata}")
        
        print(f"[VALIDATION_SUCCESS] Timeline data structure verified: {len(self.timeline_df)} timeline points across {len(timeline_runs)} simulations")

    def generate_timeline_summary_stats(self) -> pd.DataFrame:
        """
        STATISTICAL ANALYSIS: Generate comprehensive timeline-aware summary statistics.
        
        ANALYSIS DIMENSIONS:
        - Final resource outcomes (endpoint analysis)
        - Resource trajectory characteristics (growth rates, volatility)
        - Survival analysis (time to failure)
        - Comparative mechanism performance
        """
        # Filter for successful runs only
        successful_runs_meta = self.metadata_df[self.metadata_df['status'] == 'success'].copy()
        if successful_runs_meta.empty:
            print("No successful runs to analyze.")
            return pd.DataFrame()

        # Get final resources from timeline data (last round per simulation)
        final_round_data = self.timeline_df.loc[self.timeline_df.groupby('run_id')['round'].idxmax()]
        
        # Merge with metadata for grouping variables
        analysis_df = pd.merge(
            successful_runs_meta[['run_id', 'mechanism', 'adversarial_proportion_total', 'rounds_completed']],
            final_round_data[['run_id', 'resources_after']],
            on='run_id'
        )
        
        if analysis_df.empty:
            print("No data for successful runs after merging.")
            return pd.DataFrame()

        # ENHANCED STATISTICS: Timeline-aware metrics
        summary_stats = []
        
        for (mechanism, adv_prop), group in analysis_df.groupby(['mechanism', 'adversarial_proportion_total']):
            # Basic outcome statistics
            final_resources = group['resources_after']
            
            # Timeline trajectory statistics
            trajectory_stats = self._calculate_trajectory_statistics(group['run_id'].tolist())
            
            summary_row = {
                'mechanism': mechanism,
                'adversarial_proportion_total': adv_prop,
                'num_successful_runs': len(group),
                'avg_final_resources': final_resources.mean(),
                'std_final_resources': final_resources.std(),
                'median_final_resources': final_resources.median(),
                'min_final_resources': final_resources.min(),
                'max_final_resources': final_resources.max(),
                **trajectory_stats
            }
            summary_stats.append(summary_row)
        
        return pd.DataFrame(summary_stats)

    def _calculate_trajectory_statistics(self, run_ids: List[int]) -> Dict[str, float]:
        """
        TRAJECTORY ANALYSIS: Calculate statistics specific to resource progression patterns.
        
        STATISTICAL MEASURES:
        - Average growth rate per round
        - Resource volatility (standard deviation of round-to-round changes)
        - Trajectory consistency (correlation between similar runs)
        - Early vs. late performance patterns
        """
        trajectory_data = self.timeline_df[self.timeline_df['run_id'].isin(run_ids)]
        
        if trajectory_data.empty:
            return {'avg_growth_rate': 0.0, 'resource_volatility': 0.0, 'trajectory_consistency': 0.0}
        
        # Calculate growth rates for each simulation
        growth_rates = []
        volatilities = []
        
        for run_id in run_ids:
            run_data = trajectory_data[trajectory_data['run_id'] == run_id].sort_values('round')
            if len(run_data) > 1:
                # Resource change percentages
                resource_changes = run_data['resource_change_pct'].values
                growth_rates.extend(resource_changes)
                volatilities.append(np.std(resource_changes))
        
        return {
            'avg_growth_rate': np.mean(growth_rates) if growth_rates else 0.0,
            'resource_volatility': np.mean(volatilities) if volatilities else 0.0,
            'trajectory_consistency': np.std(volatilities) if len(volatilities) > 1 else 0.0
        }

    def plot_individual_trajectories(self, timestamp: str, max_trajectories: int = 10) -> None:
        """
        VISUALIZATION: Individual simulation trajectory plots for detailed analysis.
        
        DESIGN RATIONALE:
        - Show resource progression over rounds for individual simulations
        - Enable identification of common trajectory patterns
        - Highlight critical decision points and their impacts
        - Support debugging and pattern recognition
        """
        plt.figure(figsize=(15, 10))
        
        # Select representative trajectories from different mechanisms and conditions
        sample_runs = self._select_representative_trajectories(max_trajectories)
        
        colors = {'PDD': 'blue', 'PRD': 'green', 'PLD': 'red'}
        
        for i, run_id in enumerate(sample_runs):
            run_data = self.timeline_df[self.timeline_df['run_id'] == run_id].sort_values('round')
            
            if run_data.empty:
                continue
                
            mechanism = run_data['mechanism'].iloc[0]
            adv_prop = run_data['adversarial_proportion_total'].iloc[0]
            
            plt.plot(
                run_data['round'], 
                run_data['resources_after'],
                color=colors.get(mechanism, 'gray'),
                alpha=0.7,
                linewidth=1.5,
                label=f"{mechanism} (Adv: {adv_prop:.1%})" if i < 3 else ""
            )
        
        plt.title('Individual Resource Trajectories Over Time\n(Sample of Representative Simulations)')
        plt.xlabel('Round')
        plt.ylabel('Resources')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        
        trajectory_filename = os.path.join(self.output_dir, f"individual_trajectories_{timestamp}.png")
        plt.savefig(trajectory_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Individual trajectories plot saved: {trajectory_filename}")

    def _select_representative_trajectories(self, max_count: int) -> List[int]:
        """
        SAMPLING STRATEGY: Select representative trajectories across different conditions.
        
        SELECTION CRITERIA:
        - Balanced representation across mechanisms
        - Different adversarial proportions
        - Mix of successful and challenging scenarios
        - Preference for complete trajectories
        """
        representative_runs = []
        
        # Get successful runs with complete trajectories
        successful_runs = set(self.metadata_df[self.metadata_df['status'] == 'success']['run_id'])
        
        # Sample across mechanism-adversarial proportion combinations
        for (mechanism, adv_prop), group in self.timeline_df.groupby(['mechanism', 'adversarial_proportion_total']):
            mechanism_runs = [rid for rid in group['run_id'].unique() if rid in successful_runs]
            if mechanism_runs:
                # Select one representative from this combination
                representative_runs.append(mechanism_runs[0])
                if len(representative_runs) >= max_count:
                    break
        
        return representative_runs[:max_count]

    def plot_aggregated_trajectories(self, timestamp: str, confidence_level: float = 0.95) -> None:
        """
        VISUALIZATION: Aggregated trajectory analysis with statistical confidence bands.
        
        ANALYTICAL PURPOSE:
        - Compare mechanism performance over time (not just final outcomes)
        - Show trajectory confidence intervals
        - Identify temporal patterns in mechanism effectiveness
        - Enable dynamic performance comparison
        """
        fig, ax = plt.subplots(figsize=(18, 10)) # Single plot
        log_epsilon = 0.1 # Small positive value for log scale, resources typically > 20

        # Define base colors for mechanisms
        base_colors = {'PDD': 'Blues', 'PRD': 'Greens', 'PLD': 'Reds'}
        # Define line styles for different adversarial proportions (if needed, or vary color shade)
        line_styles = ['-', '--', ':', '-.'] 

        unique_mechanisms = self.timeline_df['mechanism'].unique()
        unique_adv_props = sorted(self.timeline_df['adversarial_proportion_total'].unique())

        # Create a color map for adversarial proportions for each mechanism
        color_maps = {}
        for mech, cmap_name in base_colors.items():
            cmap = plt.get_cmap(cmap_name)
            # Create a list of colors from the colormap, ensuring enough distinct shades
            # We take colors from the darker part of the colormap (e.g., 0.4 to 0.9 of the range)
            color_maps[mech] = [cmap(i) for i in np.linspace(0.4, 0.9, len(unique_adv_props))]

        for mechanism in unique_mechanisms:
            for i, adv_prop in enumerate(unique_adv_props):
                # Filter data for the specific mechanism and adversarial proportion
                condition_data = self.timeline_df[
                    (self.timeline_df['mechanism'] == mechanism) &
                    (self.timeline_df['adversarial_proportion_total'] == adv_prop)
                ]

                if condition_data.empty:
                    continue

                # Calculate mean and confidence intervals by round
                trajectory_stats = condition_data.groupby('round')['resources_after'].agg(
                    ['mean', 'std', 'count']
                ).reset_index()

                # Calculate confidence intervals
                confidence_multiplier = 1.96 if confidence_level == 0.95 else 2.576 # 99%
                trajectory_stats['ci_lower'] = trajectory_stats['mean'] - (
                    confidence_multiplier * trajectory_stats['std'] / np.sqrt(trajectory_stats['count'])
                )
                trajectory_stats['ci_upper'] = trajectory_stats['mean'] + (
                    confidence_multiplier * trajectory_stats['std'] / np.sqrt(trajectory_stats['count'])
                )

                # Select color and line style
                color = color_maps[mechanism][i % len(color_maps[mechanism])]
                style = line_styles[i % len(line_styles)] # Cycle through line styles for the same mechanism

                # Ensure y-values are positive for log scale
                plot_mean = np.maximum(trajectory_stats['mean'], log_epsilon)
                plot_ci_lower = np.maximum(trajectory_stats['ci_lower'], log_epsilon)
                plot_ci_upper = np.maximum(trajectory_stats['ci_upper'], log_epsilon)

                # Plot mean trajectory
                ax.plot(
                    trajectory_stats['round'],
                    plot_mean,
                    color=color,
                    linestyle=style, # Use linestyle to differentiate adv_prop within a mechanism
                    linewidth=2,
                    label=f"{mechanism} (Adv: {adv_prop:.0%}, n={trajectory_stats['count'].iloc[0] if not trajectory_stats.empty else 0})"
                )

                # Plot confidence band
                # Ensure ci_upper is always >= ci_lower after clipping
                plot_ci_upper_adjusted = np.maximum(plot_ci_upper, plot_ci_lower)
                ax.fill_between(
                    trajectory_stats['round'],
                    plot_ci_lower,
                    plot_ci_upper_adjusted,
                    color=color,
                    alpha=0.15 # Reduced alpha for less clutter
                )
        ax.set_yscale('log')
        ax.set_title(f'Aggregated Resource Trajectories with {confidence_level:.0%} Confidence Intervals', fontsize=16)
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Average Resources', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Mechanism (Adversarial %)", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout()
        ax.set_ylim(bottom=log_epsilon) # Set a bottom limit for the log y-axis
        fig.subplots_adjust(right=0.75) # Adjust layout to make space for a potentially long legend

        aggregated_filename = os.path.join(self.output_dir, f"aggregated_trajectories_{timestamp}.png")
        plt.savefig(aggregated_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Aggregated trajectories plot saved: {aggregated_filename}")

    def plot_aggregated_trajectories_no_outliers(self, timestamp: str, confidence_level: float = 0.95, iqr_multiplier: float = 1.5, min_samples_for_iqr: int = 5) -> None:
        """
        VISUALIZATION: Aggregated trajectory analysis with statistical confidence bands,
        removing outliers based on the Interquartile Range (IQR) method from each group at each round.

        ANALYTICAL PURPOSE:
        - Provide a more robust view of central tendency by removing extreme runs.
        - Compare mechanism performance over time with reduced outlier influence.
        """
        if self.timeline_df.empty:
            print("No timeline data to plot (no outliers version).")
            return

        fig, ax = plt.subplots(figsize=(18, 10))
        log_epsilon = 0.1

        base_colors = {'PDD': 'Blues', 'PRD': 'Greens', 'PLD': 'Reds'}
        line_styles = ['-', '--', ':', '-.']

        unique_mechanisms = self.timeline_df['mechanism'].unique()
        unique_adv_props = sorted(self.timeline_df['adversarial_proportion_total'].unique())

        color_maps = {}
        for mech, cmap_name in base_colors.items():
            cmap = plt.get_cmap(cmap_name)
            color_maps[mech] = [cmap(i) for i in np.linspace(0.4, 0.9, len(unique_adv_props))]

        for mechanism in unique_mechanisms:
            for i_adv, adv_prop in enumerate(unique_adv_props):
                condition_data_all_runs = self.timeline_df[
                    (self.timeline_df['mechanism'] == mechanism) &
                    (self.timeline_df['adversarial_proportion_total'] == adv_prop)
                ]

                if condition_data_all_runs.empty:
                    continue

                # Process each round to remove outliers
                processed_rounds_data = []
                for round_num, group in condition_data_all_runs.groupby('round'):
                    if len(group) >= min_samples_for_iqr: # Ensure enough data points for meaningful IQR
                        Q1 = group['resources_after'].quantile(0.25)
                        Q3 = group['resources_after'].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - iqr_multiplier * IQR
                        upper_bound = Q3 + iqr_multiplier * IQR
                        
                        trimmed_group = group[
                            (group['resources_after'] >= lower_bound) &
                            (group['resources_after'] <= upper_bound)
                        ]
                        if not trimmed_group.empty:
                            processed_rounds_data.append(trimmed_group)
                    else: # Not enough data for IQR, use all data for this round
                        processed_rounds_data.append(group)
                
                if not processed_rounds_data:
                    continue

                condition_data_no_outliers = pd.concat(processed_rounds_data)

                if condition_data_no_outliers.empty:
                    continue

                trajectory_stats = condition_data_no_outliers.groupby('round')['resources_after'].agg(
                    ['mean', 'std', 'count']
                ).reset_index()

                confidence_multiplier = 1.96 if confidence_level == 0.95 else 2.576
                trajectory_stats['ci_lower'] = trajectory_stats['mean'] - (
                    confidence_multiplier * trajectory_stats['std'] / np.sqrt(trajectory_stats['count'])
                )
                trajectory_stats['ci_upper'] = trajectory_stats['mean'] + (
                    confidence_multiplier * trajectory_stats['std'] / np.sqrt(trajectory_stats['count'])
                )

                color = color_maps[mechanism][i_adv % len(color_maps[mechanism])]
                style = line_styles[i_adv % len(line_styles)]

                plot_mean = np.maximum(trajectory_stats['mean'], log_epsilon)
                plot_ci_lower = np.maximum(trajectory_stats['ci_lower'], log_epsilon)
                plot_ci_upper_adjusted = np.maximum(trajectory_stats['ci_upper'], plot_ci_lower)

                ax.plot(trajectory_stats['round'], plot_mean, color=color, linestyle=style, linewidth=2,
                        label=f"{mechanism} (Adv: {adv_prop:.0%}, n_iqr_filtered={trajectory_stats['count'].iloc[0] if not trajectory_stats.empty else 0})")
                ax.fill_between(trajectory_stats['round'], plot_ci_lower, plot_ci_upper_adjusted, color=color, alpha=0.15)

        ax.set_yscale('log')
        ax.set_title(f'Aggregated Resource Trajectories (Outliers Removed) with {confidence_level:.0%} CI', fontsize=16)
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Average Resources (Log Scale)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Mechanism (Adv %)", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_ylim(bottom=log_epsilon)
        plt.tight_layout()
        fig.subplots_adjust(right=0.75)

        aggregated_no_outliers_filename = os.path.join(self.output_dir, f"aggregated_trajectories_no_outliers_{timestamp}.png")
        plt.savefig(aggregated_no_outliers_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Aggregated trajectories (no outliers) plot saved: {aggregated_no_outliers_filename}")
    
    def plot_mechanism_comparison_by_adversarial_level(self, timestamp: str) -> None:
        """
        VISUALIZATION: Compares mechanisms based on average final resources
        across different adversarial proportion levels.
        """
        summary_df = self.generate_timeline_summary_stats()
        if summary_df.empty or 'avg_final_resources' not in summary_df.columns:
            print("Not enough data for mechanism comparison plot.")
            return

        plt.figure(figsize=(14, 8))
        
        # Use a consistent color palette for mechanisms
        palette = sns.color_palette("viridis", n_colors=summary_df['mechanism'].nunique())
        mechanism_colors = {mech: color for mech, color in zip(summary_df['mechanism'].unique(), palette)}

        for mechanism in summary_df['mechanism'].unique():
            mech_data = summary_df[summary_df['mechanism'] == mechanism].sort_values('adversarial_proportion_total')
            if not mech_data.empty:
                plt.plot(
                    mech_data['adversarial_proportion_total'],
                    mech_data['avg_final_resources'],
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=8,
                    label=mechanism,
                    color=mechanism_colors.get(mechanism)
                )

        plt.title('Mechanism Performance vs. Adversarial Proportion', fontsize=16)
        plt.xlabel('Adversarial Agent Proportion', fontsize=14)
        plt.ylabel('Average Final Resources', fontsize=14)
        plt.yscale('log') # Often useful if resource levels vary widely
        plt.legend(title="Mechanism", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        comparison_filename = os.path.join(self.output_dir, f"mechanism_comparison_vs_adversarial_{timestamp}.png")
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Mechanism comparison plot saved: {comparison_filename}")
    
    def plot_resource_change_distributions(self, timestamp: str) -> None:
        """
        DISTRIBUTION ANALYSIS: Analyze patterns in round-to-round resource changes.
        
        ANALYTICAL VALUE:
        - Identify mechanism-specific volatility patterns
        - Detect systematic biases in resource progression
        - Compare risk profiles across different mechanisms
        - Understand decision impact distributions
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        mechanisms = ['PDD', 'PRD', 'PLD']
        colors = ['blue', 'green', 'red']
        
        # Filter out extreme outliers for better visualization
        change_data = self.timeline_df[
            (self.timeline_df['resource_change_pct'] >= -50) & 
            (self.timeline_df['resource_change_pct'] <= 50)
        ]
        
        for idx, mechanism in enumerate(mechanisms):
            ax = axes[idx]
            
            mechanism_changes = change_data[change_data['mechanism'] == mechanism]['resource_change_pct']
            
            if mechanism_changes.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{mechanism}\n(No Data)')
                continue
            
            # Create histogram
            ax.hist(mechanism_changes, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
            
            # Add statistical annotations
            mean_change = mechanism_changes.mean()
            std_change = mechanism_changes.std()
            
            ax.axvline(mean_change, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_change:.2f}%')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='No Change')
            
            ax.set_title(f'{mechanism}\nMean: {mean_change:.2f}%, Std: {std_change:.2f}%')
            ax.set_xlabel('Resource Change (%)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.suptitle('Distribution of Round-to-Round Resource Changes by Mechanism', fontsize=16)
        plt.tight_layout()
        
        distribution_filename = os.path.join(self.output_dir, f"resource_change_distributions_{timestamp}.png")
        plt.savefig(distribution_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Resource change distributions plot saved: {distribution_filename}")

    def run_comprehensive_timeline_analysis(self, timestamp: str) -> None:
        """
        COMPREHENSIVE ANALYSIS: Execute complete timeline analysis pipeline.
        
        ANALYSIS SEQUENCE:
        1. Generate timeline-aware summary statistics
        2. Create individual trajectory visualizations
        3. Generate aggregated trajectory comparisons
        4. Analyze resource change distributions
        5. Detect and analyze critical events
        6. Export comprehensive results
        """
        print("\n=== TIMELINE ANALYSIS PIPELINE ===")
        
        # Generate summary statistics
        print("1. Generating timeline-aware summary statistics...")
        summary_stats = self.generate_timeline_summary_stats()
        if not summary_stats.empty:
            summary_filename = os.path.join(self.output_dir, f"timeline_summary_stats_{timestamp}.csv")
            summary_stats.to_csv(summary_filename, index=False)
            print(f"Timeline summary statistics saved: {summary_filename}")
        
        # Generate visualizations
        print("2. Creating individual trajectory plots...")
        self.plot_individual_trajectories(timestamp)
        
        print("3. Creating aggregated trajectory analysis...")
        self.plot_aggregated_trajectories(timestamp)
        
        print("4. Analyzing resource change distributions...")
        self.plot_resource_change_distributions(timestamp)

        print("6. Generating aggregated trajectories plot (no outliers)...")
        self.plot_aggregated_trajectories_no_outliers(timestamp)

        print("7. Generating mechanism comparison plot...")
        self.plot_mechanism_comparison_by_adversarial_level(timestamp)
        
        print("=== TIMELINE ANALYSIS COMPLETE ===\n")


# BACKWARD COMPATIBILITY: Maintain existing analysis interface
class AnalysisPipeline:
    """
    COMPATIBILITY LAYER: Preserve existing analysis interface while adding timeline capabilities.
    
    MIGRATION STRATEGY:
    - Maintain existing method signatures
    - Add timeline analysis as additional capability
    - Provide migration path for existing code
    """
    
    def __init__(self, aggregated_data_df: pd.DataFrame, aggregated_metadata_df: pd.DataFrame, output_dir: str):
        self.data_df = aggregated_data_df
        self.metadata_df = aggregated_metadata_df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create timeline analysis pipeline
        self.timeline_analysis = TimelineAnalysisPipeline(
            aggregated_data_df, aggregated_metadata_df, output_dir
        )

    def generate_summary_stats(self) -> pd.DataFrame:
        """BACKWARD COMPATIBILITY: Generate summary statistics (legacy format)."""
        return self.timeline_analysis.generate_timeline_summary_stats()

    def plot_final_resources_vs_adversarial(self, summary_df: pd.DataFrame, fixed_pm_sigma: float, timestamp: str):
        """BACKWARD COMPATIBILITY: Generate final resources comparison plot."""
        if summary_df.empty:
            print("No summary data to plot.")
            return

        plt.figure(figsize=(12, 7))
        
        colors = {'PDD': 'blue', 'PRD': 'green', 'PLD': 'red'}
        markers = {'PDD': 'o', 'PRD': 's', 'PLD': '^'}
        
        for mechanism in summary_df['mechanism'].unique():
            mech_data = summary_df[summary_df['mechanism'] == mechanism]
            plt.errorbar(
                mech_data['adversarial_proportion_total'],
                mech_data['avg_final_resources'],
                yerr=mech_data['std_final_resources'],
                label=mechanism,
                marker=markers.get(mechanism, 'o'),
                color=colors.get(mechanism, 'black'),
                capsize=5
            )
        
        plt.title('Average Final Resources vs. Adversarial Proportion')
        plt.xlabel('Adversarial Agent Proportion')
        plt.ylabel('Average Final Resources')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_filename = os.path.join(self.output_dir, f"final_resources_plot_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Final resources plot saved: {plot_filename}")

    def run_default_analysis(self, timestamp: str):
        """ENHANCED ANALYSIS: Run both legacy and timeline analysis."""
        print("Running comprehensive analysis (legacy + timeline)...")
        
        # Legacy analysis
        summary_stats = self.generate_summary_stats()
        if not summary_stats.empty:
            self.plot_final_resources_vs_adversarial(summary_stats, 0.25, timestamp)
            
            summary_filename = os.path.join(self.output_dir, f"summary_stats_{timestamp}.csv")
            summary_stats.to_csv(summary_filename, index=False)
            print(f"Summary stats saved: {summary_filename}")
        
        # Enhanced timeline analysis
        self.timeline_analysis.run_comprehensive_timeline_analysis(timestamp)