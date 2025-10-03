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

@dataclass
class AdversarialPointMetrics:
    """Data structure for storing comprehensive metrics of a specific adversarial point."""
    adversarial_proportion: float
    sample_size: int
    mechanisms_present: List[str]
    final_performance_median: Dict[str, float]
    final_performance_iqr: Dict[str, Tuple[float, float]]
    trajectory_stability: Dict[str, float]
    data_quality_score: float

class PointSpecificAdversarialAnalyzer:
    """
    Comprehensive point-specific adversarial proportion analysis system.
    
    This system moves beyond regime-based aggregation to provide precise,
    granular analysis of democratic mechanism performance at exact adversarial
    proportions, ensuring statistical rigor and preventing misleading aggregations.
    """
    
    def __init__(self, timeline_df: pd.DataFrame, output_base_dir: str, min_sample_size: int = 10):
        """
        Initialize the analyzer with comprehensive validation.
        
        Args:
            timeline_df: DataFrame with columns ['mechanism', 'adversarial_proportion_total', 
                        'round', 'resources_after', 'replication_run_index']
            output_base_dir: Base directory for hierarchical output organization
            min_sample_size: Minimum sample size for statistical validity
        """
        self._validate_input_data(timeline_df)
        self.timeline_df = timeline_df
        self.output_base_dir = Path(output_base_dir)
        self.min_sample_size = min_sample_size
        
        # Create comprehensive directory structure
        self._initialize_directory_structure()
        
        # Analyze available adversarial proportions with statistical validation
        self.adversarial_points = self._analyze_adversarial_points()
        
        # Generate comprehensive data quality report
        self._generate_data_quality_report()
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Comprehensive input data validation with detailed error reporting."""
        required_columns = {'mechanism', 'adversarial_proportion_total', 'round', 
                           'resources_after', 'replication_run_index'}
        
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and ranges
        if df['adversarial_proportion_total'].min() < 0 or df['adversarial_proportion_total'].max() > 1:
            raise ValueError("Adversarial proportions must be between 0 and 1.")
        
        if df['resources_after'].min() < 0:
            raise ValueError("Resources cannot be negative.")
        
        if df['round'].min() < 0:
            raise ValueError("Round numbers cannot be negative.")
    
    def _initialize_directory_structure(self) -> None:
        """Create hierarchical directory structure for organized output."""
        self.directories = {
            'individual_points': self.output_base_dir / 'individual_adversarial_points',
            'comparative_analysis': self.output_base_dir / 'comparative_analysis',
            'data_quality': self.output_base_dir / 'data_quality_reports',
            'statistical_summaries': self.output_base_dir / 'statistical_summaries'
        }
        
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
    
    def _analyze_adversarial_points(self) -> Dict[float, AdversarialPointMetrics]:
        """
        Comprehensive analysis of available adversarial proportions with statistical validation.
        
        Returns:
            Dictionary mapping adversarial proportions to their comprehensive metrics
        """
        adversarial_points = {}
        unique_props = sorted(self.timeline_df['adversarial_proportion_total'].unique())
        
        for prop in unique_props:
            subset = self.timeline_df[self.timeline_df['adversarial_proportion_total'] == prop]
            
            # Calculate comprehensive metrics
            final_round = subset['round'].max()
            final_performance = subset[subset['round'] == final_round]
            
            mechanisms_present = sorted(subset['mechanism'].unique())
            sample_size = len(subset['replication_run_index'].unique())
            
            # Calculate performance metrics per mechanism
            final_performance_median = {}
            final_performance_iqr = {}
            trajectory_stability = {}
            
            for mechanism in mechanisms_present:
                mech_data = final_performance[final_performance['mechanism'] == mechanism]
                if not mech_data.empty:
                    final_performance_median[mechanism] = mech_data['resources_after'].median()
                    q25 = mech_data['resources_after'].quantile(0.25)
                    q75 = mech_data['resources_after'].quantile(0.75)
                    final_performance_iqr[mechanism] = (q25, q75)
                    
                    # Calculate trajectory stability (coefficient of variation across rounds)
                    mech_trajectory = subset[subset['mechanism'] == mechanism]
                    round_medians = mech_trajectory.groupby('round')['resources_after'].median()
                    trajectory_stability[mechanism] = round_medians.std() / round_medians.mean() if round_medians.mean() > 0 else float('inf')
            
            # Calculate data quality score (0-1, higher is better)
            data_quality_score = min(1.0, sample_size / (self.min_sample_size * 2))
            
            adversarial_points[prop] = AdversarialPointMetrics(
                adversarial_proportion=prop,
                sample_size=sample_size,
                mechanisms_present=mechanisms_present,
                final_performance_median=final_performance_median,
                final_performance_iqr=final_performance_iqr,
                trajectory_stability=trajectory_stability,
                data_quality_score=data_quality_score
            )
        
        return adversarial_points
    
    def _generate_data_quality_report(self) -> None:
        """Generate comprehensive data quality assessment report."""
        report = {
            'total_adversarial_points': len(self.adversarial_points),
            'adversarial_proportions': list(self.adversarial_points.keys()),
            'data_quality_summary': {},
            'statistical_warnings': [],
            'recommendations': []
        }
        
        for prop, metrics in self.adversarial_points.items():
            report['data_quality_summary'][prop] = {
                'sample_size': metrics.sample_size,
                'mechanisms_count': len(metrics.mechanisms_present),
                'quality_score': metrics.data_quality_score,
                'sufficient_for_analysis': metrics.sample_size >= self.min_sample_size
            }
            
            if metrics.sample_size < self.min_sample_size:
                report['statistical_warnings'].append(
                    f"Adversarial proportion {prop:.1%} has only {metrics.sample_size} samples "
                    f"(minimum recommended: {self.min_sample_size})"
                )
        
        # Save data quality report
        report_path = self.directories['data_quality'] / 'comprehensive_data_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Data quality report saved: {report_path}")
        return report
    
    def generate_individual_point_analysis(self, adversarial_prop: float, timestamp: str) -> None:
        """
        Generate comprehensive analysis for a specific adversarial proportion.
        
        Args:
            adversarial_prop: Exact adversarial proportion to analyze
            timestamp: Timestamp for file naming
        """
        if adversarial_prop not in self.adversarial_points:
            raise ValueError(f"Adversarial proportion {adversarial_prop} not found in data.")
        
        metrics = self.adversarial_points[adversarial_prop]
        subset = self.timeline_df[self.timeline_df['adversarial_proportion_total'] == adversarial_prop]
        
        # Create individual point directory
        point_dir = self.directories['individual_points'] / f"adversarial_{adversarial_prop:.1%}".replace('%', 'pct')
        point_dir.mkdir(exist_ok=True)
        
        # Generate comprehensive trajectory analysis
        self._plot_individual_trajectory_analysis(subset, metrics, point_dir, timestamp)
        
        # Generate statistical summary
        self._generate_individual_statistical_summary(subset, metrics, point_dir, timestamp)
        
        # Generate mechanism comparison at this specific point
        self._plot_individual_mechanism_comparison(subset, metrics, point_dir, timestamp)
    
    def _plot_individual_trajectory_analysis(self, subset: pd.DataFrame, 
                                           metrics: AdversarialPointMetrics, 
                                           output_dir: Path, timestamp: str) -> None:
        """Generate detailed trajectory analysis for a specific adversarial proportion."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        adversarial_prop = metrics.adversarial_proportion
        
        # Plot 1: Median trajectories with confidence intervals
        for mechanism in metrics.mechanisms_present:
            mech_data = subset[subset['mechanism'] == mechanism]
            trajectory_stats = mech_data.groupby('round')['resources_after'].agg([
                'median', 'count',
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75)
            ]).reset_index()
            trajectory_stats.columns = ['round', 'median', 'count', 'q25', 'q75']
            
            ax1.plot(trajectory_stats['round'], trajectory_stats['median'], 
                    'o-', linewidth=2, markersize=4, label=f"{mechanism}")
            ax1.fill_between(trajectory_stats['round'], trajectory_stats['q25'], 
                           trajectory_stats['q75'], alpha=0.2)
        
        ax1.set_yscale('log')
        ax1.set_title(f'Performance Trajectories at {adversarial_prop:.1%} Adversarial\n'
                     f'Sample Size: {metrics.sample_size}', fontweight='bold')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Resources (Log Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance distribution
        final_round = subset['round'].max()
        final_data = subset[subset['round'] == final_round]
        
        for i, mechanism in enumerate(metrics.mechanisms_present):
            mech_final = final_data[final_data['mechanism'] == mechanism]['resources_after']
            if not mech_final.empty:
                ax2.hist(np.log10(mech_final.clip(lower=0.1)), bins=15, alpha=0.6, 
                        label=f"{mechanism} (n={len(mech_final)})")
        
        ax2.set_title(f'Final Performance Distribution\nat {adversarial_prop:.1%} Adversarial')
        ax2.set_xlabel('Log₁₀(Final Resources)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance evolution (normalized to initial)
        for mechanism in metrics.mechanisms_present:
            mech_data = subset[subset['mechanism'] == mechanism]
            trajectory_medians = mech_data.groupby('round')['resources_after'].median()
            if len(trajectory_medians) > 1:
                initial_performance = trajectory_medians.iloc[0]
                normalized_trajectory = trajectory_medians / initial_performance
                ax3.plot(trajectory_medians.index, normalized_trajectory, 
                        'o-', linewidth=2, markersize=4, label=f"{mechanism}")
        
        ax3.set_title(f'Performance Evolution (Normalized to Initial)\nat {adversarial_prop:.1%} Adversarial')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Performance Relative to Round 0')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Variability analysis
        for mechanism in metrics.mechanisms_present:
            mech_data = subset[subset['mechanism'] == mechanism]
            variability = mech_data.groupby('round')['resources_after'].agg([
                lambda x: np.percentile(x, 75) - np.percentile(x, 25)  # IQR
            ]).reset_index()
            variability.columns = ['round', 'iqr']
            
            ax4.plot(variability['round'], variability['iqr'], 
                    'o-', linewidth=2, markersize=4, label=f"{mechanism}")
        
        ax4.set_title(f'Performance Variability (IQR)\nat {adversarial_prop:.1%} Adversarial')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Interquartile Range')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = output_dir / f"comprehensive_trajectory_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Individual trajectory analysis saved: {filename}")
    
    def _generate_individual_statistical_summary(self, subset: pd.DataFrame,
                                                metrics: AdversarialPointMetrics,
                                                output_dir: Path, timestamp: str) -> None:
        """Generate detailed statistical summary for a specific adversarial proportion."""
        summary = {
            'adversarial_proportion': metrics.adversarial_proportion,
            'sample_size': metrics.sample_size,
            'data_quality_score': metrics.data_quality_score,
            'mechanisms_analyzed': metrics.mechanisms_present,
            'final_performance_statistics': {},
            'trajectory_characteristics': {},
            'statistical_tests': {}
        }
        
        final_round = subset['round'].max()
        final_data = subset[subset['round'] == final_round]
        
        for mechanism in metrics.mechanisms_present:
            mech_final = final_data[final_data['mechanism'] == mechanism]['resources_after']
            mech_full = subset[subset['mechanism'] == mechanism]
            
            if not mech_final.empty:
                summary['final_performance_statistics'][mechanism] = {
                    'median': float(mech_final.median()),
                    'mean': float(mech_final.mean()),
                    'std': float(mech_final.std()),
                    'q25': float(mech_final.quantile(0.25)),
                    'q75': float(mech_final.quantile(0.75)),
                    'min': float(mech_final.min()),
                    'max': float(mech_final.max()),
                    'sample_size': int(len(mech_final))
                }
                
                # Trajectory characteristics
                round_medians = mech_full.groupby('round')['resources_after'].median()
                summary['trajectory_characteristics'][mechanism] = {
                    'initial_performance': float(round_medians.iloc[0]),
                    'final_performance': float(round_medians.iloc[-1]),
                    'growth_ratio': float(round_medians.iloc[-1] / round_medians.iloc[0]),
                    'trajectory_stability': metrics.trajectory_stability[mechanism],
                    'rounds_analyzed': int(len(round_medians))
                }
        
        # Save statistical summary
        summary_path = output_dir / f"statistical_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Statistical summary saved: {summary_path}")
    
    def _plot_individual_mechanism_comparison(self, subset: pd.DataFrame,
                                            metrics: AdversarialPointMetrics,
                                            output_dir: Path, timestamp: str) -> None:
        """Generate mechanism comparison visualization for specific adversarial proportion."""
        if len(metrics.mechanisms_present) < 2:
            print(f"Skipping mechanism comparison (only {len(metrics.mechanisms_present)} mechanism available)")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        adversarial_prop = metrics.adversarial_proportion
        
        # Final performance comparison
        final_round = subset['round'].max()
        final_data = subset[subset['round'] == final_round]
        
        mechanisms = metrics.mechanisms_present
        final_medians = [metrics.final_performance_median[m] for m in mechanisms]
        
        bars = ax1.bar(mechanisms, final_medians, alpha=0.7, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(mechanisms)])
        ax1.set_title(f'Final Performance Comparison\nat {adversarial_prop:.1%} Adversarial')
        ax1.set_ylabel('Median Final Resources')
        ax1.set_yscale('log')
        
        # Add value labels on bars
        for bar, median in zip(bars, final_medians):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{median:.1f}', ha='center', va='bottom')
        
        # Performance growth comparison
        growth_ratios = []
        for mechanism in mechanisms:
            mech_data = subset[subset['mechanism'] == mechanism]
            round_medians = mech_data.groupby('round')['resources_after'].median()
            if len(round_medians) > 1:
                growth_ratio = round_medians.iloc[-1] / round_medians.iloc[0]
                growth_ratios.append(growth_ratio)
            else:
                growth_ratios.append(1.0)
        
        bars2 = ax2.bar(mechanisms, growth_ratios, alpha=0.7,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(mechanisms)])
        ax2.set_title(f'Performance Growth Ratio\nat {adversarial_prop:.1%} Adversarial')
        ax2.set_ylabel('Final/Initial Performance Ratio')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Growth')
        ax2.legend()
        
        # Stability comparison
        stability_scores = [metrics.trajectory_stability[m] for m in mechanisms]
        bars3 = ax3.bar(mechanisms, stability_scores, alpha=0.7,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(mechanisms)])
        ax3.set_title(f'Trajectory Stability (Lower = More Stable)\nat {adversarial_prop:.1%} Adversarial')
        ax3.set_ylabel('Coefficient of Variation')
        
        # Sample size verification
        sample_sizes = []
        for mechanism in mechanisms:
            mech_data = subset[subset['mechanism'] == mechanism]
            sample_sizes.append(len(mech_data['replication_run_index'].unique()))
        
        bars4 = ax4.bar(mechanisms, sample_sizes, alpha=0.7,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(mechanisms)])
        ax4.set_title(f'Sample Sizes by Mechanism\nat {adversarial_prop:.1%} Adversarial')
        ax4.set_ylabel('Number of Replications')
        ax4.axhline(y=self.min_sample_size, color='red', linestyle='--', 
                   alpha=0.5, label=f'Minimum ({self.min_sample_size})')
        ax4.legend()
        
        plt.tight_layout()
        
        filename = output_dir / f"mechanism_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Mechanism comparison saved: {filename}")
    
    def generate_comparative_overview(self, timestamp: str) -> None:
        """Generate comprehensive comparative analysis across all adversarial points."""
        # Performance landscape heatmap
        self._plot_performance_landscape_heatmap(timestamp)
        
        # Point-to-point performance comparison
        self._plot_point_to_point_comparison(timestamp)
        
        # Statistical significance analysis
        self._generate_statistical_significance_analysis(timestamp)
    
    def _plot_performance_landscape_heatmap(self, timestamp: str) -> None:
        """Create detailed performance landscape across all adversarial points."""
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        adversarial_props = sorted(self.adversarial_points.keys())
        
        fig, axes = plt.subplots(1, len(mechanisms), figsize=(6*len(mechanisms), 8))
        if len(mechanisms) == 1:
            axes = [axes]
        
        for mech_idx, mechanism in enumerate(mechanisms):
            # Create performance matrix for this mechanism
            mech_data = self.timeline_df[self.timeline_df['mechanism'] == mechanism]
            
            # Create heatmap data
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
                
                # Set ticks to show actual values
                x_ticks = range(0, len(heatmap_data.index), max(1, len(heatmap_data.index)//10))
                axes[mech_idx].set_xticks(x_ticks)
                axes[mech_idx].set_xticklabels([heatmap_data.index[i] for i in x_ticks])
                
                y_ticks = range(len(heatmap_data.columns))
                axes[mech_idx].set_yticks(y_ticks)
                axes[mech_idx].set_yticklabels([f"{prop:.1%}" for prop in heatmap_data.columns])
                
                plt.colorbar(im, ax=axes[mech_idx], shrink=0.6, 
                           label='Log₁₀(Median Resources)')
        
        plt.tight_layout()
        
        filename = self.directories['comparative_analysis'] / f"performance_landscape_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance landscape saved: {filename}")
    
    def _plot_point_to_point_comparison(self, timestamp: str) -> None:
        """Create point-to-point performance comparison across adversarial proportions."""
        adversarial_props = sorted(self.adversarial_points.keys())
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Final performance across adversarial points
        for mechanism in mechanisms:
            final_performances = []
            error_bars = []
            
            for prop in adversarial_props:
                if mechanism in self.adversarial_points[prop].final_performance_median:
                    median = self.adversarial_points[prop].final_performance_median[mechanism]
                    q25, q75 = self.adversarial_points[prop].final_performance_iqr[mechanism]
                    final_performances.append(median)
                    error_bars.append([median - q25, q75 - median])
                else:
                    final_performances.append(np.nan)
                    error_bars.append([0, 0])
            
            valid_indices = ~np.isnan(final_performances)
            if np.any(valid_indices):
                valid_props = np.array(adversarial_props)[valid_indices]
                valid_performances = np.array(final_performances)[valid_indices]
                valid_errors = np.array(error_bars)[valid_indices].T
                
                ax1.errorbar(valid_props, valid_performances, yerr=valid_errors,
                           fmt='o-', linewidth=2, markersize=6, label=mechanism, capsize=4)
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Adversarial Proportion')
        ax1.set_ylabel('Final Median Resources (Log Scale)')
        ax1.set_title('Final Performance vs Adversarial Pressure', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample size validation
        sample_sizes = [self.adversarial_points[prop].sample_size for prop in adversarial_props]
        bars = ax2.bar(range(len(adversarial_props)), sample_sizes, alpha=0.7)
        ax2.set_xticks(range(len(adversarial_props)))
        ax2.set_xticklabels([f"{prop:.1%}" for prop in adversarial_props], rotation=45)
        ax2.set_ylabel('Sample Size')
        ax2.set_title('Sample Size Distribution', fontweight='bold')
        ax2.axhline(y=self.min_sample_size, color='red', linestyle='--', 
                   alpha=0.7, label=f'Minimum ({self.min_sample_size})')
        ax2.legend()
        
        # Color bars based on adequacy
        for i, (bar, size) in enumerate(zip(bars, sample_sizes)):
            if size < self.min_sample_size:
                bar.set_color('red')
                bar.set_alpha(0.5)
        
        # Plot 3: Data quality scores
        quality_scores = [self.adversarial_points[prop].data_quality_score for prop in adversarial_props]
        bars3 = ax3.bar(range(len(adversarial_props)), quality_scores, alpha=0.7, color='green')
        ax3.set_xticks(range(len(adversarial_props)))
        ax3.set_xticklabels([f"{prop:.1%}" for prop in adversarial_props], rotation=45)
        ax3.set_ylabel('Data Quality Score (0-1)')
        ax3.set_title('Data Quality Assessment', fontweight='bold')
        ax3.set_ylim(0, 1)
        
        # Plot 4: Mechanism coverage
        coverage_matrix = np.zeros((len(adversarial_props), len(mechanisms)))
        for i, prop in enumerate(adversarial_props):
            for j, mechanism in enumerate(mechanisms):
                if mechanism in self.adversarial_points[prop].mechanisms_present:
                    coverage_matrix[i, j] = 1
        
        im4 = ax4.imshow(coverage_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax4.set_title('Mechanism Coverage Matrix', fontweight='bold')
        ax4.set_xlabel('Adversarial Proportion')
        ax4.set_ylabel('Mechanism')
        ax4.set_xticks(range(len(adversarial_props)))
        ax4.set_xticklabels([f"{prop:.1%}" for prop in adversarial_props], rotation=45)
        ax4.set_yticks(range(len(mechanisms)))
        ax4.set_yticklabels(mechanisms)
        
        plt.tight_layout()
        
        filename = self.directories['comparative_analysis'] / f"point_to_point_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Point-to-point comparison saved: {filename}")
    
    def _generate_statistical_significance_analysis(self, timestamp: str) -> None:
        """Generate statistical significance analysis between adversarial points."""
        from scipy import stats
        
        adversarial_props = sorted(self.adversarial_points.keys())
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        
        analysis_results = {
            'pairwise_comparisons': {},
            'anova_results': {},
            'correlation_analysis': {}
        }
        
        # Perform pairwise statistical tests
        final_round = self.timeline_df['round'].max()
        final_data = self.timeline_df[self.timeline_df['round'] == final_round]
        
        for mechanism in mechanisms:
            mechanism_results = {'mann_whitney_tests': {}, 'effect_sizes': {}}
            
            mech_final = final_data[final_data['mechanism'] == mechanism]
            
            # Pairwise Mann-Whitney U tests (non-parametric)
            for i, prop1 in enumerate(adversarial_props):
                for j, prop2 in enumerate(adversarial_props[i+1:], i+1):
                    data1 = mech_final[mech_final['adversarial_proportion_total'] == prop1]['resources_after']
                    data2 = mech_final[mech_final['adversarial_proportion_total'] == prop2]['resources_after']
                    
                    if len(data1) >= 5 and len(data2) >= 5:  # Minimum for meaningful test
                        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        # Calculate effect size (Cliff's delta approximation)
                        n1, n2 = len(data1), len(data2)
                        effect_size = (statistic - (n1 * n2) / 2) / (n1 * n2)
                        
                        mechanism_results['mann_whitney_tests'][f"{prop1:.1%}_vs_{prop2:.1%}"] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                        
                        mechanism_results['effect_sizes'][f"{prop1:.1%}_vs_{prop2:.1%}"] = float(effect_size)
            
            analysis_results['pairwise_comparisons'][mechanism] = mechanism_results
        
        # Save statistical analysis
        analysis_path = self.directories['statistical_summaries'] / f"statistical_significance_{timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"Statistical significance analysis saved: {analysis_path}")
    
    def run_complete_analysis(self, timestamp: str) -> None:
        """
        Execute complete point-specific adversarial analysis workflow.
        
        This method orchestrates the entire analysis pipeline, generating
        individual point analyses, comparative overviews, and statistical
        summaries while maintaining data integrity and statistical rigor.
        """
        print("Starting comprehensive point-specific adversarial analysis...")
        print(f"Total adversarial proportions to analyze: {len(self.adversarial_points)}")
        
        # Generate individual point analyses
        print("\nGenerating individual point analyses...")
        for prop in self.adversarial_points.keys():
            print(f"  Analyzing {prop:.1%} adversarial proportion...")
            self.generate_individual_point_analysis(prop, timestamp)
        
        # Generate comparative overview
        print("\nGenerating comparative overview...")
        self.generate_comparative_overview(timestamp)
        
        # Generate summary index
        self._generate_analysis_index(timestamp)
        
        print(f"\nComplete analysis finished. Results saved in: {self.output_base_dir}")
    
    def _generate_analysis_index(self, timestamp: str) -> None:
        """Generate HTML index for easy navigation of results."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Point-Specific Adversarial Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .adversarial-point {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; }}
                .warning {{ color: #d9534f; font-weight: bold; }}
                .good {{ color: #5cb85c; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Point-Specific Adversarial Analysis Results</h1>
                <p>Generated: {timestamp}</p>
                <p>Total Adversarial Proportions Analyzed: {len(self.adversarial_points)}</p>
            </div>
            
            <div class="section">
                <h2>Individual Adversarial Point Analyses</h2>
        """
        
        for prop, metrics in self.adversarial_points.items():
            quality_class = "good" if metrics.data_quality_score >= 0.5 else "warning"
            html_content += f"""
                <div class="adversarial-point">
                    <h3>{prop:.1%} Adversarial Proportion</h3>
                    <p>Sample Size: <span class="{quality_class}">{metrics.sample_size}</span></p>
                    <p>Data Quality Score: <span class="{quality_class}">{metrics.data_quality_score:.2f}</span></p>
                    <p>Mechanisms: {', '.join(metrics.mechanisms_present)}</p>
                    <p><a href="individual_adversarial_points/adversarial_{prop:.1%}/".replace('%', 'pct')>View Detailed Analysis</a></p>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Comparative Analyses</h2>
                <ul>
                    <li><a href="comparative_analysis/">Cross-Point Comparisons</a></li>
                    <li><a href="statistical_summaries/">Statistical Significance Tests</a></li>
                    <li><a href="data_quality_reports/">Data Quality Assessment</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        index_path = self.output_base_dir / "analysis_index.html"
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        print(f"Analysis index generated: {index_path}")

# Usage example and integration with existing system
def run_point_specific_analysis(timeline_df: pd.DataFrame, output_dir: str, timestamp: str) -> None:
    """
    Main entry point for point-specific adversarial analysis.
    
    This function provides a clean interface for integrating the point-specific
    analysis system with existing visualization workflows.
    """
    try:
        analyzer = PointSpecificAdversarialAnalyzer(
            timeline_df=timeline_df,
            output_base_dir=output_dir,
            min_sample_size=10  # Configurable based on research requirements
        )
        
        analyzer.run_complete_analysis(timestamp)
        
        print("Point-specific adversarial analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in point-specific analysis: {e}")
        raise