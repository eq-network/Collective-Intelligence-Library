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
            'adversarial_proportion_total', 'replication_run_index'
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
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        # Get unique adversarial proportions for subplot organization
        adv_props = sorted(self.timeline_df['adversarial_proportion_total'].unique())
        
        colors = {'PDD': 'blue', 'PRD': 'green', 'PLD': 'red'}
        
        for idx, adv_prop in enumerate(adv_props[:4]):  # Limit to 4 subplots
            ax = axes[idx]
            
            for mechanism in ['PDD', 'PRD', 'PLD']:
                mechanism_data = self.timeline_df[
                    (self.timeline_df['mechanism'] == mechanism) & 
                    (self.timeline_df['adversarial_proportion_total'] == adv_prop)
                ]
                
                if mechanism_data.empty:
                    continue
                
                # Calculate mean and confidence intervals by round
                trajectory_stats = mechanism_data.groupby('round')['resources_after'].agg([
                    'mean', 'std', 'count'
                ]).reset_index()
                
                # Calculate confidence intervals
                confidence_multiplier = 1.96 if confidence_level == 0.95 else 2.576  # 99%
                trajectory_stats['ci_lower'] = trajectory_stats['mean'] - (
                    confidence_multiplier * trajectory_stats['std'] / np.sqrt(trajectory_stats['count'])
                )
                trajectory_stats['ci_upper'] = trajectory_stats['mean'] + (
                    confidence_multiplier * trajectory_stats['std'] / np.sqrt(trajectory_stats['count'])
                )
                
                # Plot mean trajectory
                ax.plot(
                    trajectory_stats['round'], 
                    trajectory_stats['mean'],
                    color=colors[mechanism],
                    linewidth=2,
                    label=f"{mechanism} (n={trajectory_stats['count'].iloc[0]})"
                )
                
                # Plot confidence band
                ax.fill_between(
                    trajectory_stats['round'],
                    trajectory_stats['ci_lower'],
                    trajectory_stats['ci_upper'],
                    color=colors[mechanism],
                    alpha=0.2
                )
            
            ax.set_title(f'Adversarial Proportion: {adv_prop:.1%}')
            ax.set_xlabel('Round')
            ax.set_ylabel('Average Resources')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
        
        # Remove unused subplots
        for idx in range(len(adv_props), 4):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'Aggregated Resource Trajectories with {confidence_level:.0%} Confidence Intervals', fontsize=16)
        plt.tight_layout()
        
        aggregated_filename = os.path.join(self.output_dir, f"aggregated_trajectories_{timestamp}.png")
        plt.savefig(aggregated_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Aggregated trajectories plot saved: {aggregated_filename}")

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

    def detect_critical_events(self, threshold_pct: float = 10.0) -> pd.DataFrame:
        """
        EVENT DETECTION: Identify significant resource changes for pattern analysis.
        
        DETECTION CRITERIA:
        - Resource changes exceeding threshold percentage
        - Consecutive significant changes (momentum patterns)
        - Mechanism-specific event frequency analysis
        - Adversarial condition correlation
        """
        critical_events = self.timeline_df[
            abs(self.timeline_df['resource_change_pct']) >= threshold_pct
        ].copy()
        
        if critical_events.empty:
            return pd.DataFrame()
        
        # Classify event types
        critical_events['event_type'] = critical_events['resource_change_pct'].apply(
            lambda x: 'Large Gain' if x >= threshold_pct else 'Large Loss'
        )
        
        # Add context information
        critical_events['event_magnitude'] = abs(critical_events['resource_change_pct'])
        
        return critical_events[['run_id', 'round', 'mechanism', 'adversarial_proportion_total', 
                              'event_type', 'event_magnitude', 'chosen_portfolio', 'decision_idx']]

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
        
        # Critical event analysis
        print("5. Detecting critical events...")
        critical_events = self.detect_critical_events()
        if not critical_events.empty:
            events_filename = os.path.join(self.output_dir, f"critical_events_{timestamp}.csv")
            critical_events.to_csv(events_filename, index=False)
            print(f"Critical events analysis saved: {events_filename}")
            print(f"   Detected {len(critical_events)} critical events across all simulations")
        
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