# Essential imports for enhanced trajectory visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

class HeatmapVisualizer:
    def __init__(self, timeline_df: pd.DataFrame, output_dir: str):
        """
        Initializes the visualizer with timeline data and an output directory.

        Args:
            timeline_df: Pandas DataFrame containing the simulation timeline data.
                         Expected columns: 'mechanism', 'adversarial_proportion_total',
                                           'round', 'resources_after'.
            output_dir: Path to the directory where plots will be saved.
        """
        if not isinstance(timeline_df, pd.DataFrame):
            raise ValueError("timeline_df must be a pandas DataFrame.")
        if not timeline_df.empty and not {'mechanism', 'adversarial_proportion_total', 'round', 'resources_after'}.issubset(timeline_df.columns):
            raise ValueError("timeline_df is missing one or more required columns: 'mechanism', 'adversarial_proportion_total', 'round', 'resources_after'")
        
        self.timeline_df = timeline_df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_enhanced_trajectories(self, timestamp: str, confidence_level: float = 0.95) -> None:
        """
        Enhanced trajectory visualization using medians and strategic visual organization.
        
        Key improvements:
        - Uses median instead of mean for robustness
        - Faceted by mechanism for clarity  
        - Strategic selection of adversarial proportions
        - Cleaner confidence intervals
        """
        if self.timeline_df.empty:
            print("Timeline data is empty. Skipping enhanced trajectories plot.")
            return

        # Strategic selection of adversarial proportions for clarity
        available_adv_props = sorted(self.timeline_df['adversarial_proportion_total'].unique())
        
        target_props = [0.1, 0.3, 0.5, 0.7, 0.9]
        selected_adv_props = [prop for prop in target_props if prop in available_adv_props]
        
        if 0.0 in available_adv_props and 0.0 not in selected_adv_props:
            selected_adv_props = [0.0] + selected_adv_props
        selected_adv_props.sort() # Ensure sorted order
        
        if not selected_adv_props: # If no target_props were found and 0.0 wasn't available
            selected_adv_props = available_adv_props # Fallback to all available if selection is empty

        if not selected_adv_props:
            print("No adversarial proportions to plot. Skipping enhanced trajectories.")
            return

        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        if not mechanisms:
            print("No mechanisms found in data. Skipping enhanced trajectories.")
            return
            
        fig, axes = plt.subplots(1, len(mechanisms), figsize=(6*len(mechanisms), 8), sharey=True, squeeze=False)
        axes = axes.flatten() # Ensure axes is always an array
        
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
                
                ax.plot(
                    trajectory_stats['round'],
                    plot_median,
                    color=color,
                    linewidth=3,
                    label=f"Adv: {adv_prop:.0%} (n={trajectory_stats['count'].iloc[0] if len(trajectory_stats) > 0 else 0})",
                    alpha=0.9
                )
                
                ax.fill_between(
                    trajectory_stats['round'],
                    plot_q1,
                    plot_q3,
                    color=color,
                    alpha=0.2
                )
            
            ax.set_yscale('log')
            ax.set_title(f'{mechanism} Performance Trajectories', fontsize=14, fontweight='bold')
            ax.set_xlabel('Round', fontsize=12)
            if mech_idx == 0:
                ax.set_ylabel('Median Resources (Log Scale)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(title="Adversarial %", frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(bottom=log_epsilon)
        
        plt.tight_layout()
        enhanced_filename = os.path.join(self.output_dir, f"enhanced_trajectories_{timestamp}.png")
        plt.savefig(enhanced_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced trajectories plot saved: {enhanced_filename}")

    def plot_trajectory_heatmaps(self, timestamp: str) -> None:
        """
        Revolutionary visualization: Heatmaps revealing the hidden topography of democratic mechanism performance.
        """
        if self.timeline_df.empty:
            print("Timeline data is empty. Skipping trajectory heatmaps.")
            return

        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        if not mechanisms:
            print("No mechanisms found in data. Skipping trajectory heatmaps.")
            return

        fig, axes = plt.subplots(2, len(mechanisms), figsize=(8*len(mechanisms), 12), squeeze=False)
        axes = axes.reshape(2, -1) # Ensure axes is always 2D
        
        for mech_idx, mechanism in enumerate(mechanisms):
            mech_data = self.timeline_df[self.timeline_df['mechanism'] == mechanism]
            if mech_data.empty:
                axes[0, mech_idx].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, mech_idx].transAxes)
                axes[1, mech_idx].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[1, mech_idx].transAxes)
                axes[0, mech_idx].set_title(f'{mechanism}: Median Performance Landscape\n(No Data)', fontsize=14, fontweight='bold')
                axes[1, mech_idx].set_title(f'{mechanism}: Resilience Topography\n(No Data)', fontsize=14, fontweight='bold')
                continue

            heatmap_data = mech_data.groupby(['round', 'adversarial_proportion_total'])['resources_after'].median().unstack()
            if heatmap_data.empty:
                axes[0, mech_idx].text(0.5, 0.5, 'No Data for Heatmap', ha='center', va='center', transform=axes[0, mech_idx].transAxes)
                axes[1, mech_idx].text(0.5, 0.5, 'No Data for Heatmap', ha='center', va='center', transform=axes[1, mech_idx].transAxes)
                axes[0, mech_idx].set_title(f'{mechanism}: Median Performance Landscape\n(No Data)', fontsize=14, fontweight='bold')
                axes[1, mech_idx].set_title(f'{mechanism}: Resilience Topography\n(No Data)', fontsize=14, fontweight='bold')
                continue

            ax_raw = axes[0, mech_idx]
            heatmap_data_log = np.log10(heatmap_data.clip(lower=0.1))
            
            im1 = ax_raw.imshow(heatmap_data_log.T, aspect='auto', cmap='plasma', origin='lower')
            ax_raw.set_title(f'{mechanism}: Median Performance Landscape\n(Log₁₀ Resources)', fontsize=14, fontweight='bold')
            ax_raw.set_xlabel('Round', fontsize=12)
            ax_raw.set_ylabel('Adversarial Proportion', fontsize=12)
            
            ax_raw.set_xticks(range(0, len(heatmap_data.index), 5))
            ax_raw.set_xticklabels(heatmap_data.index[::5])
            ax_raw.set_yticks(range(len(heatmap_data.columns)))
            ax_raw.set_yticklabels([f"{prop:.0%}" for prop in heatmap_data.columns])
            
            cbar1 = plt.colorbar(im1, ax=ax_raw, shrink=0.6)
            cbar1.set_label('Log₁₀(Median Resources)', rotation=270, labelpad=20)
            
            ax_resilience = axes[1, mech_idx]
            
            if 0.0 not in heatmap_data.columns:
                ax_resilience.text(0.5, 0.5, 'Baseline (0% Adv) Missing', ha='center', va='center', transform=ax_resilience.transAxes)
                ax_resilience.set_title(f'{mechanism}: Resilience Topography\n(Baseline Missing)', fontsize=14, fontweight='bold')
                continue

            baseline_performance = heatmap_data.loc[:, 0.0]
            resilience_data = heatmap_data.div(baseline_performance, axis=0)
            resilience_clipped = resilience_data.clip(0.01, 2.0)
            
            im2 = ax_resilience.imshow(resilience_clipped.T, aspect='auto', cmap='RdYlBu_r', origin='lower', vmin=0.1, vmax=1.5)
            ax_resilience.set_title(f'{mechanism}: Resilience Topography\n(Relative to 0% Adversarial)', fontsize=14, fontweight='bold')
            ax_resilience.set_xlabel('Round', fontsize=12)
            ax_resilience.set_ylabel('Adversarial Proportion', fontsize=12)
            
            ax_resilience.set_xticks(range(0, len(heatmap_data.index), 5))
            ax_resilience.set_xticklabels(heatmap_data.index[::5])
            ax_resilience.set_yticks(range(len(heatmap_data.columns)))
            ax_resilience.set_yticklabels([f"{prop:.0%}" for prop in heatmap_data.columns])
            
            cbar2 = plt.colorbar(im2, ax=ax_resilience, shrink=0.6)
            cbar2.set_label('Resilience Ratio\n(Performance/Baseline)', rotation=270, labelpad=25)
            
            contour_levels = [0.5, 0.75, 1.0]
            ax_resilience.contour(resilience_clipped.T, levels=contour_levels, colors='white', alpha=0.6, linewidths=1)
        
        plt.tight_layout()
        heatmap_filename = os.path.join(self.output_dir, f"trajectory_heatmaps_{timestamp}.png")
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trajectory heatmaps saved: {heatmap_filename}")

    def analyze_phase_transitions(self, timestamp: str) -> dict:
        """
        Detect and quantify phase transitions in mechanism performance using median-based analysis.
        """
        if self.timeline_df.empty:
            print("Timeline data is empty. Skipping phase transition analysis.")
            return {}

        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        phase_analysis = {}
        
        for mechanism in mechanisms:
            mech_data = self.timeline_df[self.timeline_df['mechanism'] == mechanism]
            if mech_data.empty:
                phase_analysis[mechanism] = {"status": "No data for this mechanism"}
                continue

            heatmap_data = mech_data.groupby(['round', 'adversarial_proportion_total'])['resources_after'].median().unstack()
            if heatmap_data.empty or heatmap_data.shape[0] < 2 or heatmap_data.shape[1] < 2: # Need at least 2x2 for gradient
                phase_analysis[mechanism] = {"status": "Not enough data points for gradient analysis"}
                continue

            adv_gradient = np.gradient(heatmap_data.values, axis=1)
            time_gradient = np.gradient(heatmap_data.values, axis=0)
            
            max_adv_change_idx = np.unravel_index(np.argmax(np.abs(adv_gradient)), adv_gradient.shape)
            max_time_change_idx = np.unravel_index(np.argmax(np.abs(time_gradient)), time_gradient.shape)
            
            analysis_results = {
                'critical_adversarial_threshold_by_gradient': float(heatmap_data.columns[max_adv_change_idx[1]]),
                'critical_time_point_by_gradient': int(heatmap_data.index[max_time_change_idx[0]]),
                'performance_at_50_percent_adversarial': None
            }

            if 0.5 in heatmap_data.columns:
                analysis_results['performance_at_50_percent_adversarial'] = float(heatmap_data.loc[:, 0.5].median())

            if 0.0 in heatmap_data.columns:
                baseline = heatmap_data.loc[:, 0.0]
                resilience_matrix = heatmap_data.div(baseline, axis=0)
                if not resilience_matrix.empty:
                    resilience_stability = resilience_matrix.var(axis=0)
                    if not resilience_stability.empty:
                        analysis_results['most_unstable_adversarial_level'] = float(resilience_stability.idxmax())
                        analysis_results['resilience_stability_profile'] = resilience_stability.apply(float).to_dict()
            
            phase_analysis[mechanism] = analysis_results
        
        analysis_filename = os.path.join(self.output_dir, f"phase_transition_analysis_{timestamp}.json")
        import json
        with open(analysis_filename, 'w') as f:
            json.dump(phase_analysis, f, indent=2) # Removed default=float, handled by converting values
        
        print(f"Phase transition analysis saved: {analysis_filename}")
        return phase_analysis

    def plot_comparative_resilience(self, timestamp: str) -> None:
        """
        Focus on the key research question: How do mechanisms degrade under adversarial pressure?
        """
        if self.timeline_df.empty:
            print("Timeline data is empty. Skipping comparative resilience plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        if not mechanisms:
            print("No mechanisms found in data. Skipping comparative resilience plot.")
            plt.close(fig)
            return
            
        colors = {'PDD': '#1f77b4', 'PLD': '#ff7f0e', 'PRD': '#2ca02c', 'Default': '#7f7f7f'}
        
        final_round = self.timeline_df['round'].max()
        final_performance = self.timeline_df[self.timeline_df['round'] == final_round].groupby(
            ['mechanism', 'adversarial_proportion_total']
        )['resources_after'].median().reset_index()
        
        if final_performance.empty:
            print("No final round performance data. Skipping comparative resilience plot.")
            plt.close(fig)
            return

        baselines = final_performance[final_performance['adversarial_proportion_total'] == 0.0].set_index('mechanism')['resources_after']
        
        for mechanism in mechanisms:
            mech_data = final_performance[final_performance['mechanism'] == mechanism]
            if mech_data.empty:
                continue
            
            current_color = colors.get(mechanism, colors['Default'])
            ax1.plot(mech_data['adversarial_proportion_total'], 
                    mech_data['resources_after'], 
                    'o-', color=current_color, linewidth=2, markersize=6, label=mechanism)
            
            if mechanism in baselines:
                baseline = baselines[mechanism]
                normalized_performance = mech_data['resources_after'] / baseline
                ax2.plot(mech_data['adversarial_proportion_total'], 
                        normalized_performance, 
                        'o-', color=current_color, linewidth=2, markersize=6, label=mechanism)
            else:
                print(f"Warning: Baseline (0% adv) not found for mechanism {mechanism}. Skipping normalized plot for it.")
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Adversarial Proportion', fontsize=12)
        ax1.set_ylabel('Final Median Resources (Log Scale)', fontsize=12)
        ax1.set_title('Absolute Performance vs. Adversarial Pressure', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (0% Adversarial)')
        ax2.set_xlabel('Adversarial Proportion', fontsize=12)
        ax2.set_ylabel('Normalized Performance\n(Relative to 0% Adversarial)', fontsize=12)
        ax2.set_title('Mechanism Resilience Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        resilience_filename = os.path.join(self.output_dir, f"resilience_analysis_{timestamp}.png")
        plt.savefig(resilience_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Resilience analysis saved: {resilience_filename}")

    def plot_adversarial_regime_analysis(self, timestamp: str) -> None:
        """
        Revolutionary regime-based analysis: Transform continuous adversarial pressure into 
        discrete operational climates, revealing how democratic mechanisms adapt to 
        fundamentally different environmental conditions.
        
        This approach recognizes that democratic systems likely operate in qualitatively 
        different modes depending on adversarial intensity:
        - Cooperative Regime (0-25%): Near-optimal information aggregation
        - Contested Regime (25-50%): Competitive but functional dynamics  
        - Hostile Regime (50-75%): Stressed coordination under pressure
        - Survival Regime (75-100%): Crisis-mode resilience testing
        """
        if self.timeline_df.empty:
            print("Timeline data is empty. Skipping adversarial regime analysis.")
            return

        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        if not mechanisms:
            print("No mechanisms found in data. Skipping adversarial regime analysis.")
            return

        # Define adversarial regimes with meaningful labels
        regime_definitions = [
            (0.0, 0.25, "Cooperative\n(0-25% Adversarial)", '#2E8B57'),      # Sea Green
            (0.25, 0.5, "Contested\n(25-50% Adversarial)", '#FF8C00'),       # Dark Orange  
            (0.5, 0.75, "Hostile\n(50-75% Adversarial)", '#DC143C'),         # Crimson
            (0.75, 1.0, "Survival\n(75-100% Adversarial)", '#8B0000')        # Dark Red
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
                # Calculate median performance across all runs in this regime
                regime_summary = regime_subset.groupby(['mechanism', 'round'])['resources_after'].median().reset_index()
                regime_summary['regime'] = regime_name
                regime_summary['regime_color'] = regime_color
                regime_summary['regime_bounds'] = f"{min_adv:.0%}-{max_adv:.0%}"
                regime_data.append(regime_summary)
        
        if not regime_data:
            print("No data found for any adversarial regime. Skipping regime analysis.")
            return
        
        regime_df = pd.concat(regime_data, ignore_index=True)
        
        # Create 2x2 subplot grid for four regimes
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Democratic Mechanisms Across Adversarial Regimes\nMedian Performance Landscapes', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        
        # Colors for mechanisms (consistent across all plots)
        mechanism_colors = {
            'PDD': '#1f77b4',   # Blue
            'PLD': '#ff7f0e',   # Orange  
            'PRD': '#2ca02c',   # Green
        }
        
        # Default color for unknown mechanisms
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
            
            # Plot each mechanism's trajectory in this regime
            for mechanism in mechanisms:
                mech_data = regime_subset[regime_subset['mechanism'] == mechanism]
                if mech_data.empty:
                    continue
                
                # Get mechanism color
                if mechanism in mechanism_colors:
                    color = mechanism_colors[mechanism]
                else:
                    color = default_colors[color_idx % len(default_colors)]
                    color_idx += 1
                
                # Plot trajectory with enhanced styling
                ax.plot(mech_data['round'], mech_data['resources_after'], 
                    'o-', color=color, linewidth=3, markersize=5, 
                    alpha=0.9, label=mechanism, markerfacecolor='white', 
                    markeredgecolor=color, markeredgewidth=2)
            
            # Customize subplot
            ax.set_yscale('log')
            ax.set_title(regime_name, fontsize=14, fontweight='bold', color=regime_color)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Median Resources (Log Scale)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
            
            # Add regime-specific background tinting
            ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.05, color=regime_color)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        regime_filename = os.path.join(self.output_dir, f"adversarial_regime_analysis_{timestamp}.png")
        plt.savefig(regime_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Adversarial regime analysis saved: {regime_filename}")


    def plot_regime_performance_summary(self, timestamp: str) -> None:
        """
        Complementary analysis: Statistical summary of mechanism performance across regimes.
        
        Creates box plots and statistical summaries that quantify the performance 
        distributions within each adversarial regime, revealing not just typical 
        performance but also the variability and outlier patterns that characterize 
        each operational environment.
        """
        if self.timeline_df.empty:
            print("Timeline data is empty. Skipping regime performance summary.")
            return

        mechanisms = sorted(self.timeline_df['mechanism'].unique())
        if not mechanisms:
            print("No mechanisms found in data. Skipping regime performance summary.")
            return

        # Define the same regimes as before
        regime_definitions = [
            (0.0, 0.25, "Cooperative", '#2E8B57'),
            (0.25, 0.5, "Contested", '#FF8C00'),
            (0.5, 0.75, "Hostile", '#DC143C'),
            (0.75, 1.0, "Survival", '#8B0000')
        ]
        
        # Prepare data for box plot analysis
        regime_performance_data = []
        
        for min_adv, max_adv, regime_name, regime_color in regime_definitions:
            regime_subset = self.timeline_df[
                (self.timeline_df['adversarial_proportion_total'] >= min_adv) & 
                (self.timeline_df['adversarial_proportion_total'] < max_adv if max_adv < 1.0 else 
                self.timeline_df['adversarial_proportion_total'] <= max_adv)
            ]
            
            if not regime_subset.empty:
                # Get final round performance for statistical summary 
                final_round = regime_subset['round'].max()
                final_performance = regime_subset[regime_subset['round'] == final_round]
                
                for mechanism in mechanisms:
                    mech_final = final_performance[final_performance['mechanism'] == mechanism]
                    if not mech_final.empty:
                        for _, row in mech_final.iterrows():
                            regime_performance_data.append({
                                'regime': regime_name,
                                'mechanism': mechanism,
                                'final_resources': row['resources_after'],
                                'regime_color': regime_color
                            })
        
        if not regime_performance_data:
            print("No final performance data found for regime summary.")
            return
            
        regime_perf_df = pd.DataFrame(regime_performance_data)
        
        # Create summary visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Statistical Performance Summary Across Adversarial Regimes', 
                    fontsize=16, fontweight='bold')
        
        # Left plot: Box plots showing performance distributions
        import seaborn as sns
        sns.boxplot(data=regime_perf_df, x='regime', y='final_resources', hue='mechanism', ax=ax1)
        ax1.set_yscale('log')
        ax1.set_title('Performance Distribution by Regime', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Adversarial Regime', fontsize=12)
        ax1.set_ylabel('Final Resources (Log Scale)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Right plot: Regime comparison heatmap
        pivot_data = regime_perf_df.groupby(['regime', 'mechanism'])['final_resources'].median().unstack()
        if not pivot_data.empty:
            # Normalize by the best performance in each regime for comparative analysis
            normalized_pivot = pivot_data.div(pivot_data.max(axis=1), axis=0)
            
            im = ax2.imshow(normalized_pivot.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            ax2.set_title('Relative Performance Heatmap\n(Normalized by Best in Regime)', 
                        fontsize=14, fontweight='bold')
            ax2.set_xlabel('Mechanism', fontsize=12)
            ax2.set_ylabel('Adversarial Regime', fontsize=12)
            
            # Set ticks and labels
            ax2.set_xticks(range(len(pivot_data.columns)))
            ax2.set_xticklabels(pivot_data.columns)
            ax2.set_yticks(range(len(pivot_data.index)))
            ax2.set_yticklabels(pivot_data.index)
            
            # Add text annotations showing actual values
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    if not np.isnan(normalized_pivot.iloc[i, j]):
                        ax2.text(j, i, f'{normalized_pivot.iloc[i, j]:.2f}', 
                            ha='center', va='center', fontweight='bold',
                            color='white' if normalized_pivot.iloc[i, j] < 0.5 else 'black')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('Relative Performance\n(1.0 = Best in Regime)', rotation=270, labelpad=25)
        
        plt.tight_layout()
        
        summary_filename = os.path.join(self.output_dir, f"regime_performance_summary_{timestamp}.png")
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Regime performance summary saved: {summary_filename}")