# Enhanced imports for point-specific adversarial analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

class EnhancedHeatmapVisualizer:
    """
    Enhanced visualization system that combines regime-based analysis with precise
    point-specific adversarial proportion analysis.
    
    This system addresses the fundamental limitation of broad regime groupings
    by providing granular analysis at exact adversarial proportions while
    maintaining backward compatibility with existing visualization workflows.
    """
    
    def __init__(self, timeline_df: pd.DataFrame, output_dir: str, min_sample_size: int = 10):
        """
        Initialize enhanced visualizer with comprehensive validation and setup.
        
        Args:
            timeline_df: Pandas DataFrame containing simulation timeline data
            output_dir: Base directory for organized output structure
            min_sample_size: Minimum sample size for statistical validity warnings
        """
        if not isinstance(timeline_df, pd.DataFrame):
            raise ValueError("timeline_df must be a pandas DataFrame.")
        
        required_columns = {'mechanism', 'adversarial_proportion_total', 'round', 'resources_after'}
        if not timeline_df.empty and not required_columns.issubset(timeline_df.columns):
            missing = required_columns - set(timeline_df.columns)
            raise ValueError(f"timeline_df is missing required columns: {missing}")
        
        self.timeline_df = timeline_df
        self.output_dir = Path(output_dir)
        self.min_sample_size = min_sample_size
        
        # Create comprehensive directory structure
        self._initialize_enhanced_directory_structure()
        
        # Analyze available adversarial proportions with statistical validation
        self.adversarial_points = self._analyze_available_adversarial_points()
        
        # Generate data quality assessment
        self.data_quality_report = self._generate_data_quality_assessment()
    
    def _initialize_enhanced_directory_structure(self) -> None:
        """Create hierarchical directory structure for organized analysis output."""
        self.directories = {
            'legacy_visualizations': self.output_dir / 'legacy',
            'point_specific': self.output_dir / 'point_analysis',
            'individual_points': self.output_dir / 'point_analysis' / 'adv_points',
            'comparative_analysis': self.output_dir / 'point_analysis' / 'compare',
            'data_quality': self.output_dir / 'dq',
            'statistical_validation': self.output_dir / 'stats_val'
        }
        
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
    
    def _analyze_available_adversarial_points(self) -> Dict[float, Dict]:
        """
        Comprehensive analysis of available adversarial proportions.
        
        Returns detailed metrics for each adversarial proportion including
        sample sizes, mechanism coverage, and statistical adequacy.
        """
        if self.timeline_df.empty:
            return {}
        
        adversarial_points = {}
        unique_props = sorted(self.timeline_df['adversarial_proportion_total'].unique())
        
        for prop in unique_props:
            subset = self.timeline_df[self.timeline_df['adversarial_proportion_total'] == prop]
            
            # Calculate comprehensive metrics
            final_round = subset['round'].max() if not subset.empty else 0
            final_performance = subset[subset['round'] == final_round] if not subset.empty else pd.DataFrame()
            
            mechanisms_present = sorted(subset['mechanism'].unique()) if not subset.empty else []
            
            # Calculate sample size (unique replications)
            if 'replication_run_index' in subset.columns:
                sample_size = len(subset['replication_run_index'].unique())
            else:
                # Fallback: estimate from data structure
                sample_size = len(subset) // (len(mechanisms_present) * (final_round + 1)) if mechanisms_present and final_round > 0 else len(subset)
            
            # Statistical adequacy assessment
            statistical_adequacy = sample_size >= self.min_sample_size
            data_quality_score = min(1.0, sample_size / (self.min_sample_size * 2))
            
            # Performance metrics per mechanism
            final_performance_metrics = {}
            for mechanism in mechanisms_present:
                mech_data = final_performance[final_performance['mechanism'] == mechanism] if not final_performance.empty else pd.DataFrame()
                if not mech_data.empty and 'resources_after' in mech_data.columns:
                    resources = mech_data['resources_after']
                    final_performance_metrics[mechanism] = {
                        'median': float(resources.median()),
                        'q25': float(resources.quantile(0.25)),
                        'q75': float(resources.quantile(0.75)),
                        'count': int(len(resources))
                    }
            
            adversarial_points[prop] = {
                'adversarial_proportion': prop,
                'sample_size': sample_size,
                'mechanisms_present': mechanisms_present,
                'final_performance_metrics': final_performance_metrics,
                'statistical_adequacy': statistical_adequacy,
                'data_quality_score': data_quality_score,
                'rounds_available': int(final_round + 1) if final_round >= 0 else 0
            }
        
        return adversarial_points
    
    def _generate_data_quality_assessment(self) -> Dict:
        """Generate comprehensive data quality assessment report."""
        report = {
            'total_adversarial_points': len(self.adversarial_points),
            'adversarial_proportions_available': list(self.adversarial_points.keys()),
            'statistical_warnings': [],
            'data_coverage_analysis': {},
            'recommended_analysis_points': []
        }
        
        # Assess each adversarial point
        for prop, metrics in self.adversarial_points.items():
            report['data_coverage_analysis'][prop] = {
                'sample_size': metrics['sample_size'],
                'mechanisms_count': len(metrics['mechanisms_present']),
                'quality_score': metrics['data_quality_score'],
                'adequate_for_analysis': metrics['statistical_adequacy']
            }
            
            # Generate warnings for inadequate data
            if not metrics['statistical_adequacy']:
                report['statistical_warnings'].append(
                    f"Adversarial proportion {prop:.1%} has only {metrics['sample_size']} samples "
                    f"(minimum recommended: {self.min_sample_size})"
                )
            else:
                report['recommended_analysis_points'].append(prop)
        
        return report
    
    def generate_individual_adversarial_point_analysis(self, adversarial_prop: float, timestamp: str) -> None:
        """
        Generate comprehensive deep-dive analysis for a specific adversarial proportion.
        
        This method creates detailed visualizations and statistical summaries
        for a single adversarial proportion, avoiding the aggregation issues
        inherent in regime-based analysis.
        """
        if adversarial_prop not in self.adversarial_points:
            print(f"Warning: Adversarial proportion {adversarial_prop:.1%} not found in data.")
            return
        
        metrics = self.adversarial_points[adversarial_prop]
        subset = self.timeline_df[self.timeline_df['adversarial_proportion_total'] == adversarial_prop]
        
        if subset.empty:
            print(f"Warning: No data available for {adversarial_prop:.1%} adversarial proportion.")
            return
        
        # Create individual point directory
        point_dir = self.directories['individual_points'] / f"adv_{adversarial_prop:.1%}".replace('%', 'pct')
        point_dir.mkdir(parents=True, exist_ok=True) # Ensure all parents exist up to this point_dir
        
        print(f"Generating analysis for {adversarial_prop:.1%} adversarial proportion...")
        
        # Generate comprehensive visualizations
        self._plot_point_specific_trajectories(subset, metrics, point_dir, timestamp)
        self._plot_point_specific_mechanism_comparison(subset, metrics, point_dir, timestamp)
        self._plot_point_specific_distribution_analysis(subset, metrics, point_dir, timestamp)
        
        # Generate statistical summary
        self._generate_point_specific_statistical_summary(subset, metrics, point_dir, timestamp)
        
        print(f"Individual analysis completed for {adversarial_prop:.1%} adversarial proportion.")
    
    def _plot_point_specific_trajectories(self, subset: pd.DataFrame, metrics: Dict, 
                                        output_dir: Path, timestamp: str) -> None:
        """Generate detailed trajectory analysis for specific adversarial proportion."""
        adversarial_prop = metrics['adversarial_proportion']
        mechanisms = metrics['mechanisms_present']
        
        if not mechanisms:
            print(f"No mechanisms found for {adversarial_prop:.1%} adversarial proportion.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Trajectory plot with confidence intervals
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, mechanism in enumerate(mechanisms):
            mech_data = subset[subset['mechanism'] == mechanism]
            if mech_data.empty:
                continue
            
            # Calculate trajectory statistics
            trajectory_stats = mech_data.groupby('round')['resources_after'].agg([
                'median', 'count',
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75)
            ]).reset_index()
            trajectory_stats.columns = ['round', 'median', 'count', 'q25', 'q75']
            
            color = colors[i % len(colors)]
            
            # Main trajectory line
            ax1.plot(trajectory_stats['round'], trajectory_stats['median'], 
                    'o-', color=color, linewidth=3, markersize=5, 
                    label=f"{mechanism} (n={trajectory_stats['count'].iloc[0] if len(trajectory_stats) > 0 else 0})")
            
            # Confidence interval
            ax1.fill_between(trajectory_stats['round'], trajectory_stats['q25'], 
                           trajectory_stats['q75'], color=color, alpha=0.2)
        
        ax1.set_yscale('log')
        ax1.set_title(f'Performance Trajectories at {adversarial_prop:.1%} Adversarial\n'
                     f'Sample Size: {metrics["sample_size"]}', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Resources (Log Scale)', fontsize=12)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Performance growth analysis
        for i, mechanism in enumerate(mechanisms):
            mech_data = subset[subset['mechanism'] == mechanism]
            if mech_data.empty:
                continue
                
            trajectory_medians = mech_data.groupby('round')['resources_after'].median()
            if len(trajectory_medians) > 1:
                initial_performance = trajectory_medians.iloc[0]
                if initial_performance > 0:
                    normalized_trajectory = trajectory_medians / initial_performance
                    color = colors[i % len(colors)]
                    ax2.plot(trajectory_medians.index, normalized_trajectory, 
                            'o-', color=color, linewidth=2, markersize=4, label=mechanism)
        
        ax2.set_title(f'Normalized Performance Growth\nat {adversarial_prop:.1%} Adversarial', 
                     fontweight='bold', fontsize=14)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Performance Relative to Round 0', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Growth')
        
        # Variability analysis (IQR over time)
        for i, mechanism in enumerate(mechanisms):
            mech_data = subset[subset['mechanism'] == mechanism]
            if mech_data.empty:
                continue
                
            variability = mech_data.groupby('round')['resources_after'].agg([
                lambda x: np.percentile(x, 75) - np.percentile(x, 25)
            ]).reset_index()
            variability.columns = ['round', 'iqr']
            
            color = colors[i % len(colors)]
            ax3.plot(variability['round'], variability['iqr'], 
                    'o-', color=color, linewidth=2, markersize=4, label=mechanism)
        
        ax3.set_title(f'Performance Variability (IQR)\nat {adversarial_prop:.1%} Adversarial', 
                     fontweight='bold', fontsize=14)
        ax3.set_xlabel('Round', fontsize=12)
        ax3.set_ylabel('Interquartile Range', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Final performance distribution
        final_round = subset['round'].max()
        final_data = subset[subset['round'] == final_round]
        
        for i, mechanism in enumerate(mechanisms):
            mech_final = final_data[final_data['mechanism'] == mechanism]['resources_after']
            if not mech_final.empty:
                color = colors[i % len(colors)]
                ax4.hist(np.log10(mech_final.clip(lower=0.1)), bins=15, alpha=0.6, 
                        color=color, label=f"{mechanism} (n={len(mech_final)})", density=True)
        
        ax4.set_title(f'Final Performance Distribution\nat {adversarial_prop:.1%} Adversarial', 
                     fontweight='bold', fontsize=14)
        ax4.set_xlabel('Log₁₀(Final Resources)', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = output_dir / f"traj_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Trajectory analysis saved: {filename}")
    
    def _plot_point_specific_mechanism_comparison(self, subset: pd.DataFrame, metrics: Dict,
                                                output_dir: Path, timestamp: str) -> None:
        """Generate mechanism comparison for specific adversarial proportion."""
        adversarial_prop = metrics['adversarial_proportion']
        mechanisms = metrics['mechanisms_present']
        
        if len(mechanisms) < 2:
            print(f"  Skipping mechanism comparison (only {len(mechanisms)} mechanism available)")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        final_round = subset['round'].max()
        final_data = subset[subset['round'] == final_round]
        
        # Final performance comparison
        final_medians = []
        final_iqrs = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for mechanism in mechanisms:
            if mechanism in metrics['final_performance_metrics']:
                final_medians.append(metrics['final_performance_metrics'][mechanism]['median'])
                q25 = metrics['final_performance_metrics'][mechanism]['q25']
                q75 = metrics['final_performance_metrics'][mechanism]['q75']
                final_iqrs.append([q25, q75])
            else:
                final_medians.append(np.nan)
                final_iqrs.append([np.nan, np.nan])
        
        # Bar plot with error bars
        x_pos = np.arange(len(mechanisms))
        bars = ax1.bar(x_pos, final_medians, alpha=0.7, 
                      color=colors[:len(mechanisms)])
        
        # Add IQR error bars
        for i, (bar, (q25, q75)) in enumerate(zip(bars, final_iqrs)):
            if not np.isnan(q25) and not np.isnan(q75):
                median = final_medians[i]
                ax1.errorbar(bar.get_x() + bar.get_width()/2, median, 
                           yerr=[[median - q25], [q75 - median]], 
                           color='black', capsize=5, capthick=2)
                
                # Add value label
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{median:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_yscale('log')
        ax1.set_title(f'Final Performance Comparison\nat {adversarial_prop:.1%} Adversarial', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Mechanism', fontsize=12)
        ax1.set_ylabel('Median Final Resources (Log Scale)', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(mechanisms)
        ax1.grid(True, alpha=0.3)
        
        # Performance growth comparison
        growth_ratios = []
        for mechanism in mechanisms:
            mech_data = subset[subset['mechanism'] == mechanism]
            if not mech_data.empty:
                round_medians = mech_data.groupby('round')['resources_after'].median()
                if len(round_medians) > 1 and round_medians.iloc[0] > 0:
                    growth_ratio = round_medians.iloc[-1] / round_medians.iloc[0]
                    growth_ratios.append(growth_ratio)
                else:
                    growth_ratios.append(1.0)
            else:
                growth_ratios.append(1.0)
        
        bars2 = ax2.bar(x_pos, growth_ratios, alpha=0.7, color=colors[:len(mechanisms)])
        ax2.set_title(f'Performance Growth Ratio\nat {adversarial_prop:.1%} Adversarial', 
                     fontweight='bold', fontsize=14)
        ax2.set_xlabel('Mechanism', fontsize=12)
        ax2.set_ylabel('Final/Initial Performance Ratio', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(mechanisms)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Growth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sample size verification
        sample_sizes = []
        for mechanism in mechanisms:
            mech_data = subset[subset['mechanism'] == mechanism]
            if 'replication_run_index' in mech_data.columns:
                sample_size = len(mech_data['replication_run_index'].unique())
            else:
                sample_size = len(mech_data) // (subset['round'].nunique()) if subset['round'].nunique() > 0 else len(mech_data)
            sample_sizes.append(sample_size)
        
        bars3 = ax3.bar(x_pos, sample_sizes, alpha=0.7, color=colors[:len(mechanisms)])
        ax3.set_title(f'Sample Sizes by Mechanism\nat {adversarial_prop:.1%} Adversarial', 
                     fontweight='bold', fontsize=14)
        ax3.set_xlabel('Mechanism', fontsize=12)
        ax3.set_ylabel('Number of Replications', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(mechanisms)
        ax3.axhline(y=self.min_sample_size, color='red', linestyle='--', 
                   alpha=0.5, label=f'Minimum ({self.min_sample_size})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Color bars based on adequacy
        for bar, size in zip(bars3, sample_sizes):
            if size < self.min_sample_size:
                bar.set_color('red')
                bar.set_alpha(0.5)
        
        # Statistical significance testing (if applicable)
        if len(mechanisms) >= 2:
            pairwise_results = []
            final_data_by_mech = {}
            
            for mechanism in mechanisms:
                mech_final = final_data[final_data['mechanism'] == mechanism]['resources_after']
                if len(mech_final) >= 5:  # Minimum for meaningful statistical test
                    final_data_by_mech[mechanism] = mech_final
            
            # Perform pairwise Mann-Whitney U tests
            test_results = []
            for i, mech1 in enumerate(final_data_by_mech.keys()):
                for mech2 in list(final_data_by_mech.keys())[i+1:]:
                    try:
                        statistic, p_value = stats.mannwhitneyu(
                            final_data_by_mech[mech1], 
                            final_data_by_mech[mech2], 
                            alternative='two-sided'
                        )
                        test_results.append(f"{mech1} vs {mech2}: p={p_value:.3f}")
                    except Exception as e:
                        test_results.append(f"{mech1} vs {mech2}: Error")
            
            # Display test results
            ax4.text(0.1, 0.9, f'Statistical Significance Tests\n(Mann-Whitney U)', 
                    transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top')
            
            for i, result in enumerate(test_results[:6]):  # Limit to 6 results for space
                ax4.text(0.1, 0.8 - i*0.1, result, transform=ax4.transAxes, fontsize=10, va='top')
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
        
        plt.tight_layout()
        
        filename = output_dir / f"mech_comp_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Mechanism comparison saved: {filename}")
    
    def _plot_point_specific_distribution_analysis(self, subset: pd.DataFrame, metrics: Dict,
                                                  output_dir: Path, timestamp: str) -> None:
        """Generate distribution analysis for specific adversarial proportion."""
        adversarial_prop = metrics['adversarial_proportion']
        mechanisms = metrics['mechanisms_present']
        
        if not mechanisms:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plots of performance across rounds
        round_performance_data = []
        for mechanism in mechanisms:
            mech_data = subset[subset['mechanism'] == mechanism]
            for _, row in mech_data.iterrows():
                round_performance_data.append({
                    'mechanism': mechanism,
                    'round': row['round'],
                    'resources_after': row['resources_after']
                })
        
        if round_performance_data:
            round_df = pd.DataFrame(round_performance_data)
            
            # Box plot by mechanism
            sns.boxplot(data=round_df, x='mechanism', y='resources_after', ax=axes[0, 0])
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_title(f'Performance Distribution by Mechanism\nat {adversarial_prop:.1%} Adversarial', 
                               fontweight='bold')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Box plot by round (showing evolution)
            selected_rounds = [0, round_df['round'].max()//4, round_df['round'].max()//2, 
                             3*round_df['round'].max()//4, round_df['round'].max()]
            selected_rounds = [r for r in selected_rounds if r in round_df['round'].values]
            
            if selected_rounds:
                round_subset = round_df[round_df['round'].isin(selected_rounds)]
                sns.boxplot(data=round_subset, x='round', y='resources_after', ax=axes[0, 1])
                axes[0, 1].set_yscale('log')
                axes[0, 1].set_title(f'Performance Evolution Across Selected Rounds\nat {adversarial_prop:.1%} Adversarial', 
                                   fontweight='bold')
        
        # Performance correlation matrix (if multiple mechanisms)
        if len(mechanisms) >= 2:
            correlation_data = {}
            final_round = subset['round'].max()
            final_data = subset[subset['round'] == final_round]
            
            for mechanism in mechanisms:
                mech_final = final_data[final_data['mechanism'] == mechanism]['resources_after']
                if len(mech_final) > 0:
                    correlation_data[mechanism] = mech_final.reset_index(drop=True)
            
            if len(correlation_data) >= 2:
                # Align series to same length for correlation
                min_length = min(len(data) for data in correlation_data.values())
                aligned_data = {mech: data[:min_length] for mech, data in correlation_data.items()}
                
                corr_df = pd.DataFrame(aligned_data)
                corr_matrix = corr_df.corr()
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           ax=axes[1, 0], square=True)
                axes[1, 0].set_title(f'Performance Correlation Matrix\nat {adversarial_prop:.1%} Adversarial', 
                                   fontweight='bold')
        
        # Performance stability analysis
        stability_metrics = {}
        for mechanism in mechanisms:
            mech_data = subset[subset['mechanism'] == mechanism]
            if not mech_data.empty:
                round_medians = mech_data.groupby('round')['resources_after'].median()
                if len(round_medians) > 1:
                    cv = round_medians.std() / round_medians.mean() if round_medians.mean() > 0 else float('inf')
                    stability_metrics[mechanism] = cv
        
        if stability_metrics:
            mechanisms_list = list(stability_metrics.keys())
            stability_values = list(stability_metrics.values())
            
            bars = axes[1, 1].bar(mechanisms_list, stability_values, alpha=0.7)
            axes[1, 1].set_title(f'Trajectory Stability\n(Lower = More Stable) at {adversarial_prop:.1%} Adversarial', 
                               fontweight='bold')
            axes[1, 1].set_xlabel('Mechanism')
            axes[1, 1].set_ylabel('Coefficient of Variation')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Color bars based on stability (green = stable, red = unstable)
            for bar, value in zip(bars, stability_values):
                if value < 0.2:
                    bar.set_color('green')
                elif value > 0.5:
                    bar.set_color('red')
                else:
                    bar.set_color('orange')
        
        plt.tight_layout()
        
        filename = output_dir / f"dist_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Distribution analysis saved: {filename}")
    
    def _generate_point_specific_statistical_summary(self, subset: pd.DataFrame, metrics: Dict,
                                                    output_dir: Path, timestamp: str) -> None:
        """Generate comprehensive statistical summary for specific adversarial proportion."""
        adversarial_prop = metrics['adversarial_proportion']
        
        summary = {
            'adversarial_proportion': adversarial_prop,
            'analysis_metadata': {
                'sample_size': metrics['sample_size'],
                'data_quality_score': metrics['data_quality_score'],
                'statistical_adequacy': metrics['statistical_adequacy'],
                'rounds_analyzed': metrics['rounds_available']
            },
            'mechanisms_analyzed': metrics['mechanisms_present'],
            'performance_statistics': {},
            'trajectory_characteristics': {},
            'comparative_analysis': {}
        }
        
        # Detailed performance statistics per mechanism
        final_round = subset['round'].max()
        final_data = subset[subset['round'] == final_round]
        
        for mechanism in metrics['mechanisms_present']:
            mech_data = subset[subset['mechanism'] == mechanism]
            mech_final = final_data[final_data['mechanism'] == mechanism]['resources_after']
            
            if not mech_final.empty:
                summary['performance_statistics'][mechanism] = {
                    'final_performance': {
                        'median': float(mech_final.median()),
                        'mean': float(mech_final.mean()),
                        'std': float(mech_final.std()),
                        'q25': float(mech_final.quantile(0.25)),
                        'q75': float(mech_final.quantile(0.75)),
                        'min': float(mech_final.min()),
                        'max': float(mech_final.max()),
                        'sample_size': int(len(mech_final))
                    }
                }
                
                # Trajectory characteristics
                round_medians = mech_data.groupby('round')['resources_after'].median()
                if len(round_medians) > 1:
                    initial_performance = round_medians.iloc[0]
                    final_performance = round_medians.iloc[-1]
                    
                    summary['trajectory_characteristics'][mechanism] = {
                        'initial_performance': float(initial_performance),
                        'final_performance': float(final_performance),
                        'growth_ratio': float(final_performance / initial_performance) if initial_performance > 0 else float('inf'),
                        'trajectory_stability': float(round_medians.std() / round_medians.mean()) if round_medians.mean() > 0 else float('inf'),
                        'rounds_with_data': int(len(round_medians))
                    }
        
        # Save statistical summary
        summary_path = output_dir / f"stats_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"  Statistical summary saved: {summary_path}")
    
    def generate_all_individual_analyses(self, timestamp: str) -> None:
        """Generate individual analyses for all available adversarial proportions."""
        if not self.adversarial_points:
            print("No adversarial proportions available for analysis.")
            return
        
        print(f"Generating individual analyses for {len(self.adversarial_points)} adversarial proportions...")
        
        for prop in self.adversarial_points.keys():
            self.generate_individual_adversarial_point_analysis(prop, timestamp)
        
        print("All individual analyses completed.")
    
    def generate_comparative_cross_point_analysis(self, timestamp: str) -> None:
        """Generate comparative analysis across all adversarial proportions."""
        if len(self.adversarial_points) < 2:
            print("Insufficient adversarial proportions for comparative analysis.")
            return
        
        print("Generating comparative cross-point analysis...")
        
        # Performance landscape comparison
        self._plot_cross_point_performance_landscape(timestamp)
        
        # Point-to-point trajectory comparison
        self._plot_cross_point_trajectory_comparison(timestamp)
        
        # Statistical significance across points
        self._generate_cross_point_statistical_analysis(timestamp)
        
        print("Comparative cross-point analysis completed.")
    
    def _plot_cross_point_performance_landscape(self, timestamp: str) -> None:
        """Create performance landscape across all adversarial proportions."""
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        adversarial_props = sorted(self.adversarial_points.keys())
        
        if not mechanisms or not adversarial_props:
            print("Insufficient data for performance landscape.")
            return
        
        fig, axes = plt.subplots(1, len(mechanisms), figsize=(6*len(mechanisms), 8))
        if len(mechanisms) == 1:
            axes = [axes]
        
        for mech_idx, mechanism in enumerate(mechanisms):
            mech_data = self.timeline_df[self.timeline_df['mechanism'] == mechanism]
            
            if not mech_data.empty:
                # Create heatmap data: rounds x adversarial_proportions
                heatmap_data = mech_data.groupby(['round', 'adversarial_proportion_total'])['resources_after'].median().unstack()
                
                if not heatmap_data.empty:
                    # Use log scale for better visualization
                    heatmap_data_log = np.log10(heatmap_data.clip(lower=0.1))
                    
                    im = axes[mech_idx].imshow(heatmap_data_log.T, aspect='auto', 
                                             cmap='plasma', origin='lower')
                    
                    axes[mech_idx].set_title(f'{mechanism}: Performance Landscape\n'
                                           f'(Log₁₀ Median Resources)', fontweight='bold')
                    axes[mech_idx].set_xlabel('Round')
                    axes[mech_idx].set_ylabel('Adversarial Proportion')
                    
                    # Set ticks
                    x_ticks = range(0, len(heatmap_data.index), max(1, len(heatmap_data.index)//10))
                    axes[mech_idx].set_xticks(x_ticks)
                    axes[mech_idx].set_xticklabels([heatmap_data.index[i] for i in x_ticks])
                    
                    y_ticks = range(len(heatmap_data.columns))
                    axes[mech_idx].set_yticks(y_ticks)
                    axes[mech_idx].set_yticklabels([f"{prop:.1%}" for prop in heatmap_data.columns])
                    
                    plt.colorbar(im, ax=axes[mech_idx], shrink=0.6, 
                               label='Log₁₀(Median Resources)')
        
        plt.tight_layout()
        
        filename = self.directories['comparative_analysis'] / f"perf_landscape_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance landscape saved: {filename}")
    
    def _plot_cross_point_trajectory_comparison(self, timestamp: str) -> None:
        """Create trajectory comparison across adversarial proportions."""
        adversarial_props = sorted(self.adversarial_points.keys())
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        
        if not adversarial_props or not mechanisms:
            print("Insufficient data for trajectory comparison.")
            return
        
        fig, axes = plt.subplots(len(mechanisms), 1, figsize=(14, 6*len(mechanisms)))
        if len(mechanisms) == 1:
            axes = [axes]
        
        # Color map for adversarial proportions
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(adversarial_props)))
        
        for mech_idx, mechanism in enumerate(mechanisms):
            ax = axes[mech_idx]
            
            for prop_idx, prop in enumerate(adversarial_props):
                prop_data = self.timeline_df[
                    (self.timeline_df['mechanism'] == mechanism) &
                    (self.timeline_df['adversarial_proportion_total'] == prop)
                ]
                
                if not prop_data.empty:
                    trajectory_stats = prop_data.groupby('round')['resources_after'].agg([
                        'median', 
                        lambda x: np.percentile(x, 25),
                        lambda x: np.percentile(x, 75)
                    ]).reset_index()
                    trajectory_stats.columns = ['round', 'median', 'q25', 'q75']
                    
                    color = colors[prop_idx]
                    
                    # Main trajectory
                    ax.plot(trajectory_stats['round'], trajectory_stats['median'],
                           'o-', color=color, linewidth=2, markersize=4,
                           label=f"{prop:.1%} Adversarial")
                    
                    # Confidence interval
                    ax.fill_between(trajectory_stats['round'], trajectory_stats['q25'],
                                   trajectory_stats['q75'], color=color, alpha=0.15)
            
            ax.set_yscale('log')
            ax.set_title(f'{mechanism}: Performance Trajectories Across Adversarial Levels', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Median Resources (Log Scale)', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.directories['comparative_analysis'] / f"traj_comp_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trajectory comparison saved: {filename}")
    
    def _generate_cross_point_statistical_analysis(self, timestamp: str) -> None:
        """Generate statistical analysis across adversarial proportions."""
        adversarial_props = sorted(self.adversarial_points.keys())
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        
        analysis_results = {
            'adversarial_proportions_analyzed': adversarial_props,
            'mechanisms_analyzed': mechanisms,
            'pairwise_comparisons': {},
            'trend_analysis': {},
            'effect_size_analysis': {}
        }
        
        final_round = self.timeline_df['round'].max()
        final_data = self.timeline_df[self.timeline_df['round'] == final_round]
        
        # Pairwise statistical comparisons between adversarial proportions
        for mechanism in mechanisms:
            mechanism_results = {'mann_whitney_tests': {}, 'effect_sizes': {}}
            
            mech_final = final_data[final_data['mechanism'] == mechanism]
            
            for i, prop1 in enumerate(adversarial_props):
                for j, prop2 in enumerate(adversarial_props[i+1:], i+1):
                    data1 = mech_final[mech_final['adversarial_proportion_total'] == prop1]['resources_after']
                    data2 = mech_final[mech_final['adversarial_proportion_total'] == prop2]['resources_after']
                    
                    if len(data1) >= 5 and len(data2) >= 5:
                        try:
                            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            # Calculate effect size (Cliff's delta approximation)
                            n1, n2 = len(data1), len(data2)
                            effect_size = (statistic - (n1 * n2) / 2) / (n1 * n2)
                            
                            mechanism_results['mann_whitney_tests'][f"{prop1:.1%}_vs_{prop2:.1%}"] = {
                                'statistic': float(statistic),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05,
                                'n1': int(n1),
                                'n2': int(n2)
                            }
                            
                            mechanism_results['effect_sizes'][f"{prop1:.1%}_vs_{prop2:.1%}"] = float(effect_size)
                            
                        except Exception as e:
                            print(f"Statistical test failed for {mechanism} {prop1:.1%} vs {prop2:.1%}: {e}")
            
            analysis_results['pairwise_comparisons'][mechanism] = mechanism_results
        
        # Trend analysis (correlation with adversarial proportion)
        for mechanism in mechanisms:
            mech_final = final_data[final_data['mechanism'] == mechanism]
            
            if not mech_final.empty:
                # Calculate correlation between adversarial proportion and performance
                try:
                    correlation, p_value = stats.pearsonr(
                        mech_final['adversarial_proportion_total'],
                        mech_final['resources_after']
                    )
                    
                    analysis_results['trend_analysis'][mechanism] = {
                        'correlation_with_adversarial': float(correlation),
                        'correlation_p_value': float(p_value),
                        'correlation_significant': p_value < 0.05
                    }
                except Exception as e:
                    print(f"Trend analysis failed for {mechanism}: {e}")
        
        # Save statistical analysis
        analysis_path = self.directories['statistical_validation'] / f"cross_stats_{timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"Cross-point statistical analysis saved: {analysis_path}")
    
    def generate_data_quality_report(self, timestamp: str) -> None:
        """Generate comprehensive data quality assessment report and dashboard."""
        report_path = self.directories['data_quality'] / f"dq_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.data_quality_report, f, indent=2, default=str)
        
        # Also create a visual data quality dashboard
        self._plot_data_quality_dashboard(timestamp)
        
        print(f"Data quality report saved: {report_path}")
    
    def _plot_data_quality_dashboard(self, timestamp: str) -> None:
        """Create visual data quality assessment dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        adversarial_props = sorted(self.adversarial_points.keys())
        
        # Sample size distribution
        sample_sizes = [self.adversarial_points[prop]['sample_size'] for prop in adversarial_props]
        bars1 = ax1.bar(range(len(adversarial_props)), sample_sizes, alpha=0.7)
        ax1.set_xticks(range(len(adversarial_props)))
        ax1.set_xticklabels([f"{prop:.1%}" for prop in adversarial_props], rotation=45)
        ax1.set_ylabel('Sample Size')
        ax1.set_title('Sample Size Distribution by Adversarial Proportion', fontweight='bold')
        ax1.axhline(y=self.min_sample_size, color='red', linestyle='--', 
                   alpha=0.7, label=f'Minimum ({self.min_sample_size})')
        ax1.legend()
        
        # Color bars based on adequacy
        for bar, size in zip(bars1, sample_sizes):
            if size < self.min_sample_size:
                bar.set_color('red')
                bar.set_alpha(0.5)
            else:
                bar.set_color('green')
        
        # Data quality scores
        quality_scores = [self.adversarial_points[prop]['data_quality_score'] for prop in adversarial_props]
        bars2 = ax2.bar(range(len(adversarial_props)), quality_scores, alpha=0.7, color='blue')
        ax2.set_xticks(range(len(adversarial_props)))
        ax2.set_xticklabels([f"{prop:.1%}" for prop in adversarial_props], rotation=45)
        ax2.set_ylabel('Data Quality Score (0-1)')
        ax2.set_title('Data Quality Assessment', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Mechanism coverage heatmap
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        coverage_matrix = np.zeros((len(adversarial_props), len(mechanisms)))
        
        for i, prop in enumerate(adversarial_props):
            for j, mechanism in enumerate(mechanisms):
                if mechanism in self.adversarial_points[prop]['mechanisms_present']:
                    coverage_matrix[i, j] = 1
        
        im3 = ax3.imshow(coverage_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax3.set_title('Mechanism Coverage Matrix', fontweight='bold')
        ax3.set_xlabel('Adversarial Proportion')
        ax3.set_ylabel('Mechanism')
        ax3.set_xticks(range(len(adversarial_props)))
        ax3.set_xticklabels([f"{prop:.1%}" for prop in adversarial_props], rotation=45)
        ax3.set_yticks(range(len(mechanisms)))
        ax3.set_yticklabels(mechanisms)
        
        # Statistical adequacy summary
        adequate_counts = sum(1 for prop in adversarial_props 
                            if self.adversarial_points[prop]['statistical_adequacy'])
        inadequate_counts = len(adversarial_props) - adequate_counts
        
        ax4.pie([adequate_counts, inadequate_counts], 
               labels=['Statistically Adequate', 'Inadequate Sample Size'],
               colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax4.set_title('Statistical Adequacy Summary', fontweight='bold')
        
        plt.tight_layout()
        
        filename = self.directories['data_quality'] / f"dq_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Data quality dashboard saved: {filename}")
    
    def generate_navigation_index(self, timestamp: str) -> None:
        """Generate HTML navigation index for all analyses."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Adversarial Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .adversarial-point {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; }}
                .warning {{ color: #d9534f; font-weight: bold; }}
                .good {{ color: #5cb85c; font-weight: bold; }}
                .adequate {{ color: #5cb85c; }}
                .inadequate {{ color: #d9534f; }}
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ margin: 5px 0; }}
                a {{ text-decoration: none; color: #007acc; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced Point-Specific Adversarial Analysis Results</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Total Adversarial Proportions:</strong> {len(self.adversarial_points)}</p>
                <p><strong>Statistically Adequate Points:</strong> {sum(1 for p in self.adversarial_points.values() if p['statistical_adequacy'])}</p>
            </div>
            
            <div class="section">
                <h2>🎯 Individual Adversarial Point Analyses</h2>
                <p>Deep-dive analysis for each specific adversarial proportion, avoiding regime-based aggregation biases.</p>
        """
        
        for prop, metrics in self.adversarial_points.items():
            adequacy_class = "adequate" if metrics['statistical_adequacy'] else "inadequate"
            quality_class = "good" if metrics['data_quality_score'] >= 0.5 else "warning"
            
            html_content += f"""
                <div class="adversarial-point">
                    <h3>{prop:.1%} Adversarial Proportion</h3>
                    <p>📊 <strong>Sample Size:</strong> <span class="{adequacy_class}">{metrics['sample_size']}</span> 
                       | 📈 <strong>Quality Score:</strong> <span class="{quality_class}">{metrics['data_quality_score']:.2f}</span></p>
                    <p>🔧 <strong>Mechanisms:</strong> {', '.join(metrics['mechanisms_present'])}</p>
                    <p>📁 <a href="point_analysis/adv_points/adv_{prop:.1%}/".replace('%', 'pct')>📂 View Detailed Analysis</a></p>
                </div>
            """
        
        html_content += f"""
            </div>
            
            <div class="section">
                <h2>📈 Comparative Cross-Point Analyses</h2>
                <p>Systematic comparisons across specific adversarial proportions with statistical validation.</p>
                <ul>
                    <li>📊 <a href="point_analysis/compare/">Performance Landscape Analysis</a></li>
                    <li>📉 <a href="point_analysis/compare/">Trajectory Comparisons</a></li>
                    <li>📋 <a href="stats_val/">Statistical Significance Testing</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🔍 Data Quality Assessment</h2>
                <p>Comprehensive validation of data adequacy and statistical reliability.</p>
                <ul>
                    <li>📋 <a href="dq/">Data Quality Reports</a></li>
                    <li>📊 <a href="dq/">Coverage Analysis</a></li>
                    <li>⚠️ <a href="dq/">Statistical Adequacy Warnings</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📚 Legacy Regime-Based Analysis</h2>
                <p>Traditional regime-based visualizations for comparison purposes.</p>
                <ul>
                    <li>📊 <a href="legacy/">Enhanced Trajectories</a></li>
                    <li>🗂️ <a href="legacy/">Regime Analysis</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>⚠️ Analysis Warnings</h2>
        """
        
        if self.data_quality_report['statistical_warnings']:
            html_content += "<ul>"
            for warning in self.data_quality_report['statistical_warnings']:
                html_content += f"<li class='warning'>⚠️ {warning}</li>"
            html_content += "</ul>"
        else:
            html_content += "<p class='good'>✅ No statistical adequacy warnings detected.</p>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        index_path = self.output_dir / "analysis_index.html"
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        print(f"Navigation index generated: {index_path}")
    
    def run_complete_enhanced_analysis(self, timestamp: str) -> None:
        """
        Execute complete enhanced analysis workflow combining point-specific
        analysis with legacy regime-based visualizations.
        """
        print("🚀 Starting comprehensive enhanced adversarial analysis...")
        print(f"📊 Total adversarial proportions detected: {len(self.adversarial_points)}")
        print(f"📈 Statistically adequate points: {sum(1 for p in self.adversarial_points.values() if p['statistical_adequacy'])}")
        
        # Generate data quality report first
        print("\n📋 Generating data quality assessment...")
        self.generate_data_quality_report(timestamp)
        
        # Generate individual point-specific analyses
        print("\n🎯 Generating individual point-specific analyses...")
        self.generate_all_individual_analyses(timestamp)
        
        # Generate comparative cross-point analysis
        print("\n📈 Generating comparative cross-point analysis...")
        self.generate_comparative_cross_point_analysis(timestamp)
        
        # Generate legacy regime-based visualizations for comparison
        print("\n📚 Generating legacy regime-based visualizations...")
        self._generate_legacy_regime_visualizations(timestamp)
        
        # Generate navigation index
        print("\n📑 Generating navigation index...")
        self.generate_navigation_index(timestamp)
        
        print(f"\n✅ Complete enhanced analysis finished!")
        print(f"📁 Results organized in: {self.output_dir}")
        print(f"🌐 Open analysis_index.html for navigation")
    
    def _generate_legacy_regime_visualizations(self, timestamp: str) -> None:
        """Generate legacy regime-based visualizations for comparison purposes."""
        legacy_dir = self.directories['legacy_visualizations']
        
        # Create a temporary HeatmapVisualizer for legacy methods
        class LegacyVisualizer:
            def __init__(self, timeline_df, output_dir):
                self.timeline_df = timeline_df
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
        
        legacy_viz = LegacyVisualizer(self.timeline_df, str(legacy_dir))
        
        # Copy and adapt the legacy methods
        try:
            # Enhanced trajectories
            self._legacy_plot_enhanced_trajectories(legacy_viz, timestamp)
            
            # Adversarial regime analysis
            self._legacy_plot_adversarial_regime_analysis(legacy_viz, timestamp)
            
            print(f"Legacy regime visualizations saved in: {legacy_dir}")
            
        except Exception as e:
            print(f"Error generating legacy visualizations: {e}")
    
    def _legacy_plot_enhanced_trajectories(self, legacy_viz, timestamp: str) -> None:
        """Generate legacy enhanced trajectories plot."""
        if self.timeline_df.empty:
            return
        
        # Implement the enhanced trajectories logic from the original class
        available_adv_props = sorted(self.timeline_df['adversarial_proportion_total'].unique())
        target_props = [0.1, 0.3, 0.5, 0.7, 0.9]
        selected_adv_props = [prop for prop in target_props if prop in available_adv_props]
        
        if 0.0 in available_adv_props and 0.0 not in selected_adv_props:
            selected_adv_props = [0.0] + selected_adv_props
        selected_adv_props.sort()
        
        if not selected_adv_props:
            selected_adv_props = available_adv_props
        
        if not selected_adv_props:
            return
        
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        if not mechanisms:
            return
        
        fig, axes = plt.subplots(1, len(mechanisms), figsize=(6*len(mechanisms), 8), sharey=True, squeeze=False)
        axes = axes.flatten()
        
        log_epsilon = 0.1
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_adv_props)))
        
        for mech_idx, mechanism in enumerate(mechanisms):
            ax = axes[mech_idx]
            
            for prop_idx, adv_prop in enumerate(selected_adv_props):
                condition_data = self.timeline_df[
                    (self.timeline_df['mechanism'] == mechanism) &
                    (self.timeline_df['adversarial_proportion_total'] == adv_prop)
                ]
                
                if condition_data.empty:
                    continue
                
                trajectory_stats = condition_data.groupby('round')['resources_after'].agg([
                    'median', 
                    lambda x: np.percentile(x, 25),
                    lambda x: np.percentile(x, 75),
                    'count'
                ]).reset_index()
                
                trajectory_stats.columns = ['round', 'median', 'q1', 'q3', 'count']
                
                plot_median = np.maximum(trajectory_stats['median'], log_epsilon)
                plot_q1 = np.maximum(trajectory_stats['q1'], log_epsilon)
                plot_q3 = np.maximum(trajectory_stats['q3'], log_epsilon)
                
                color = colors[prop_idx]
                
                ax.plot(trajectory_stats['round'], plot_median, color=color, linewidth=3,
                       label=f"Adv: {adv_prop:.0%} (n={trajectory_stats['count'].iloc[0] if len(trajectory_stats) > 0 else 0})",
                       alpha=0.9)
                
                ax.fill_between(trajectory_stats['round'], plot_q1, plot_q3, color=color, alpha=0.2)
            
            ax.set_yscale('log')
            ax.set_title(f'{mechanism} Performance Trajectories', fontsize=14, fontweight='bold')
            ax.set_xlabel('Round', fontsize=12)
            if mech_idx == 0:
                ax.set_ylabel('Median Resources (Log Scale)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(title="Adversarial %", frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(bottom=log_epsilon)
        
        plt.tight_layout()
        enhanced_filename = os.path.join(legacy_viz.output_dir, f"legacy_traj_{timestamp}.png")
        plt.savefig(enhanced_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _legacy_plot_adversarial_regime_analysis(self, legacy_viz, timestamp: str) -> None:
        """Generate legacy adversarial regime analysis."""
        if self.timeline_df.empty:
            return
        
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        if not mechanisms:
            return
        
        # Define adversarial regimes
        regime_definitions = [
            (0.0, 0.25, "Cooperative\n(0-25% Adversarial)", '#2E8B57'),
            (0.25, 0.5, "Contested\n(25-50% Adversarial)", '#FF8C00'),
            (0.5, 0.75, "Hostile\n(50-75% Adversarial)", '#DC143C'),
            (0.75, 1.0, "Survival\n(75-100% Adversarial)", '#8B0000')
        ]
        
        # Create regime groupings
        regime_data = []
        for min_adv, max_adv, regime_name, regime_color in regime_definitions:
            regime_subset = self.timeline_df[
                (self.timeline_df['adversarial_proportion_total'] >= min_adv) & 
                (self.timeline_df['adversarial_proportion_total'] < max_adv if max_adv < 1.0 else 
                 self.timeline_df['adversarial_proportion_total'] <= max_adv)
            ]
            
            if not regime_subset.empty:
                regime_summary = regime_subset.groupby(['mechanism', 'round'])['resources_after'].median().reset_index()
                regime_summary['regime'] = regime_name
                regime_summary['regime_color'] = regime_color
                regime_summary['regime_bounds'] = f"{min_adv:.0%}-{max_adv:.0%}"
                regime_data.append(regime_summary)
        
        if not regime_data:
            return
        
        regime_df = pd.concat(regime_data, ignore_index=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Legacy: Democratic Mechanisms Across Adversarial Regimes\nMedian Performance Landscapes', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        
        mechanism_colors = {'PDD': '#1f77b4', 'PLD': '#ff7f0e', 'PRD': '#2ca02c'}
        default_colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        color_idx = 0
        
        for regime_idx, (min_adv, max_adv, regime_name, regime_color) in enumerate(regime_definitions):
            ax = axes[regime_idx]
            regime_subset = regime_df[regime_df['regime'] == regime_name]
            
            if regime_subset.empty:
                ax.text(0.5, 0.5, f'No Data\nfor {regime_name}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, alpha=0.6)
                ax.set_title(regime_name, fontsize=14, fontweight='bold', color=regime_color)
                continue
            
            for mechanism in mechanisms:
                mech_data = regime_subset[regime_subset['mechanism'] == mechanism]
                if mech_data.empty:
                    continue
                
                if mechanism in mechanism_colors:
                    color = mechanism_colors[mechanism]
                else:
                    color = default_colors[color_idx % len(default_colors)]
                    color_idx += 1
                
                ax.plot(mech_data['round'], mech_data['resources_after'], 
                       'o-', color=color, linewidth=3, markersize=5, 
                       alpha=0.9, label=mechanism, markerfacecolor='white', 
                       markeredgecolor=color, markeredgewidth=2)
            
            ax.set_yscale('log')
            ax.set_title(regime_name, fontsize=14, fontweight='bold', color=regime_color)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Median Resources (Log Scale)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
            ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.05, color=regime_color)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        regime_filename = os.path.join(legacy_viz.output_dir, f"legacy_regime_{timestamp}.png")
        plt.savefig(regime_filename, dpi=300, bbox_inches='tight')
        plt.close()

# Usage Integration Function
def run_enhanced_adversarial_analysis(timeline_df: pd.DataFrame, output_dir: str, timestamp: str) -> None:
    """
    Main entry point for enhanced point-specific adversarial analysis.
    
    This function provides a clean interface for integrating the enhanced
    analysis system with existing workflows while maintaining backward compatibility.
    """
    try:
        # Create enhanced visualizer
        enhanced_viz = EnhancedHeatmapVisualizer(
            timeline_df=timeline_df,
            output_dir=output_dir,
            min_sample_size=10
        )
        
        # Run complete enhanced analysis
        enhanced_viz.run_complete_enhanced_analysis(timestamp)
        
        print("\n✅ Enhanced point-specific adversarial analysis completed successfully!")
        print(f"📁 Results available at: {output_dir}")
        print(f"🌐 Open {output_dir}/analysis_index.html for navigation")
        
        return enhanced_viz
        
    except Exception as e:
        print(f"❌ Error in enhanced adversarial analysis: {e}")
        import traceback
        traceback.print_exc()
        raise