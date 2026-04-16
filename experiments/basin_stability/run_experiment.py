"""
Run the basin stability experiment.

Sweeps across:
- 3 governance mechanisms (PDD, PRD, PLD)
- 2 tracking modes (predictive λ=0.9, non-predictive λ=0.1)
- 7 adversarial fractions (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
- 100 seeds per condition (configurable)

Uses inline metric arrays filled via lax.scan for JIT-compiled execution.
Outputs CSV files for analysis and plotting.

Usage:
    python -m experiments.basin_stability.run_experiment --quick
    python -m experiments.basin_stability.run_experiment --n_seeds 100 --vmap
    python -m experiments.basin_stability.run_experiment --n_seeds 100 --vmap --plot
    python -m experiments.basin_stability.run_experiment --tracking predictive --vmap
"""
import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.basin_stability.environment import BasinStabilityEnv, run_batched
from metrics import ECONOMIC_METRICS, GOVERNANCE_METRICS
from metrics.export import write_trajectory_csv, write_summary_csv

# Default metrics for this experiment: economic + governance
BASIN_METRICS = {**ECONOMIC_METRICS, **GOVERNANCE_METRICS}

# Tracking modes: the key experimental variable
# Predictive = long-memory trust (informed delegation/election)
# Non-predictive = recency-biased trust (populism / latest-news capture)
TRACKING_MODES = {
    "predictive": 0.9,
    "non_predictive": 0.1,
}


def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def run_single(mechanism, n_agents, n_adversarial, seed, T=200,
               metrics=None, tracking_lambda=0.9, **kwargs):
    """Run a single episode. Returns final GraphState with filled metric arrays."""
    if metrics is None:
        metrics = BASIN_METRICS

    env = BasinStabilityEnv(
        mechanism=mechanism,
        n_agents=n_agents,
        n_adversarial=n_adversarial,
        tracking_lambda=tracking_lambda,
        seed=seed,
        T=T,
        metrics=metrics,
        **kwargs,
    )
    return env.run(T)


def run_sweep(
    mechanisms=("pdd", "prd", "pld"),
    tracking_modes=None,
    adversarial_fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    n_agents: int = 20,
    n_seeds: int = 100,
    T: int = 200,
    output_dir: str = None,
    metrics: dict = None,
):
    """Run the full experimental sweep (sequential) with CSV output."""
    if metrics is None:
        metrics = BASIN_METRICS
    if tracking_modes is None:
        tracking_modes = TRACKING_MODES
    metric_names = sorted(metrics.keys())

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_path = output_dir / f"trajectory_{timestamp}.csv"
    summ_path = output_dir / f"summary_{timestamp}.csv"

    results = {}

    for mechanism in mechanisms:
        for tracking_name, tracking_lambda in tracking_modes.items():
            condition_key = f"{mechanism}_{tracking_name}"
            print(f"\n{'='*50}")
            print(f"Mechanism: {mechanism.upper()} | Tracking: {tracking_name} (lambda={tracking_lambda})")
            print(f"{'='*50}")

            # Cooperative baseline (0 adversaries)
            n_baseline = min(50, n_seeds)
            baseline_evals = []
            for s in range(n_baseline):
                final = run_single(
                    mechanism, n_agents, 0, seed=s * 7919, T=T,
                    metrics=metrics, tracking_lambda=tracking_lambda,
                )
                arr = np.array(final.global_attrs["metric_resource_level"])
                eval_start = int(len(arr) * 0.75)
                log_arr = np.log(np.clip(arr[eval_start:], 1e-10, None))
                baseline_evals.append(float(np.mean(log_arr)))
            R_coop_log = float(np.median(baseline_evals))
            R_coop_display = float(np.exp(R_coop_log))
            print(f"  R_coop = {R_coop_display:.2f} (log={R_coop_log:.2f})")

            results[condition_key] = {"R_coop_log": R_coop_log, "R_coop": R_coop_display}

            for adv_frac in adversarial_fractions:
                n_adversarial = int(n_agents * adv_frac)

                final_states = []
                for seed_idx in range(n_seeds):
                    seed = seed_idx * 7919 + hash(mechanism) % 10000

                    final = run_single(
                        mechanism, n_agents, n_adversarial,
                        seed=seed, T=T, metrics=metrics,
                        tracking_lambda=tracking_lambda,
                    )
                    final_states.append(final)

                    # Stream trajectory CSV
                    run_meta = {
                        "adversarial_fraction": adv_frac,
                        "mechanism": mechanism,
                        "tracking_mode": tracking_name,
                        "tracking_lambda": tracking_lambda,
                        "seed": seed,
                    }
                    write_trajectory_csv(traj_path, final, metric_names, run_meta)

                # Compute basin stability (log-scale)
                eval_means = []
                for state in final_states:
                    arr = np.array(state.global_attrs["metric_resource_level"])
                    eval_start = int(len(arr) * 0.75)
                    log_arr = np.log(np.clip(arr[eval_start:], 1e-10, None))
                    eval_means.append(float(np.mean(log_arr)))

                n_stable = sum(1 for v in eval_means if v > R_coop_log)
                n_total = len(eval_means)
                bs = n_stable / n_total
                ci_low, ci_high = wilson_ci(n_stable, n_total)

                results[condition_key][str(adv_frac)] = {
                    "basin_stability": bs,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "n_stable": n_stable,
                    "n_total": n_total,
                }

                print(
                    f"  adv={adv_frac:.0%} -> BS={bs:.3f} "
                    f"[{ci_low:.3f}, {ci_high:.3f}]"
                )

    print(f"\nTrajectory CSV: {traj_path}")
    print(f"Summary CSV:    {summ_path}")
    return results, traj_path, summ_path


def run_sweep_vmap(
    mechanisms=("pdd", "prd", "pld"),
    tracking_modes=None,
    adversarial_fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    n_agents: int = 20,
    n_seeds: int = 100,
    T: int = 200,
    output_dir: str = None,
    metrics: dict = None,
    master_seed: int = 0,
    **kwargs,
):
    """Run the full sweep with vmap over seeds — one GPU kernel per condition."""
    if metrics is None:
        metrics = BASIN_METRICS
    if tracking_modes is None:
        tracking_modes = TRACKING_MODES
    metric_names = sorted(metrics.keys())

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_path = output_dir / f"trajectory_{timestamp}.csv"
    summ_path = output_dir / f"summary_{timestamp}.csv"

    master_key = jr.PRNGKey(master_seed)
    results = {}

    print(f"Devices: {jax.devices()}")
    print(f"vmap mode: batching {n_seeds} seeds per condition\n")

    for mechanism in mechanisms:
        for tracking_name, tracking_lambda in tracking_modes.items():
            condition_key = f"{mechanism}_{tracking_name}"
            print(f"\n{'='*50}")
            print(f"Mechanism: {mechanism.upper()} | Tracking: {tracking_name} (lambda={tracking_lambda})")
            print(f"{'='*50}")

            collapse_threshold = 20.0  # R_min
            eval_start = int(T * 0.75)
            results[condition_key] = {}

            for adv_frac in adversarial_fractions:
                n_adversarial = int(n_agents * adv_frac)

                key, subkey = jr.split(key if adv_frac > 0 else master_key)
                batch_final = run_batched(
                    mechanism, n_agents, n_adversarial, subkey, n_seeds,
                    T=T, metrics=metrics, tracking_lambda=tracking_lambda,
                    **kwargs,
                )

                # Extract resource trajectories: shape (n_seeds, T)
                resource_arrs = np.array(batch_final.global_attrs["metric_resource_level"])
                final_resources = resource_arrs[:, -1]  # (n_seeds,)

                # Key stats
                median_final_R = float(np.median(final_resources))
                mean_log_R = float(np.mean(np.log(np.clip(final_resources, 1e-10, None))))
                survival_rate = float(np.mean(final_resources > collapse_threshold))
                n_survived = int(np.sum(final_resources > collapse_threshold))

                # Selected utility over eval window
                if "metric_selected_utility" in batch_final.global_attrs:
                    util_arrs = np.array(batch_final.global_attrs["metric_selected_utility"])
                    mean_selected_yield = float(np.mean(util_arrs[:, eval_start:]))
                else:
                    mean_selected_yield = float('nan')

                ci_low, ci_high = wilson_ci(n_survived, n_seeds)

                # Write trajectory CSV
                for seed_idx in range(n_seeds):
                    metric_row = {}
                    for mn in metric_names:
                        arr = np.array(batch_final.global_attrs[f"metric_{mn}"])
                        metric_row[mn] = arr[seed_idx]
                    with open(traj_path, "a") as f:
                        if traj_path.stat().st_size == 0:
                            header = ",".join([
                                "mechanism", "tracking_mode", "tracking_lambda",
                                "adversarial_fraction", "seed", "step",
                            ] + metric_names)
                            f.write(header + "\n")
                        for step in range(T):
                            vals = [str(metric_row[mn][step]) for mn in metric_names]
                            row = (f"{mechanism},{tracking_name},{tracking_lambda},"
                                   f"{adv_frac},{seed_idx},{step}," + ",".join(vals))
                            f.write(row + "\n")

                # Write summary CSV row
                summary_metrics = {
                    "median_final_R": median_final_R,
                    "mean_log_R": mean_log_R,
                    "survival_rate": survival_rate,
                    "mean_selected_yield": mean_selected_yield,
                }
                for mn in metric_names:
                    arr = np.array(batch_final.global_attrs[f"metric_{mn}"])
                    final_vals = arr[:, -1]
                    summary_metrics[f"mean_{mn}"] = float(np.mean(final_vals))
                    summary_metrics[f"std_{mn}"] = float(np.std(final_vals))

                with open(summ_path, "a") as f:
                    if summ_path.stat().st_size == 0:
                        cols = [
                            "mechanism", "tracking_mode", "tracking_lambda",
                            "adversarial_fraction", "n_seeds",
                            "survival_rate", "survival_ci_lower", "survival_ci_upper",
                            "median_final_R", "mean_log_R", "mean_selected_yield",
                        ]
                        cols += sorted(k for k in summary_metrics if k.startswith("mean_") or k.startswith("std_"))
                        f.write(",".join(cols) + "\n")
                    vals = [mechanism, tracking_name, str(tracking_lambda),
                            str(adv_frac), str(n_seeds),
                            str(survival_rate), str(ci_low), str(ci_high),
                            str(median_final_R), str(mean_log_R), str(mean_selected_yield)]
                    vals += [str(summary_metrics[k]) for k in sorted(k for k in summary_metrics if k.startswith("mean_") or k.startswith("std_"))]
                    f.write(",".join(vals) + "\n")

                results[condition_key][str(adv_frac)] = {
                    "survival_rate": survival_rate,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "median_final_R": median_final_R,
                    "mean_log_R": mean_log_R,
                    "mean_selected_yield": mean_selected_yield,
                }

                median_log_R = float(np.log(max(median_final_R, 1e-10)))
                print(
                    f"  adv={adv_frac:.0%} | "
                    f"survival={survival_rate:.0%} ({n_survived}/{n_seeds}) | "
                    f"median log(R)={median_log_R:.1f} | "
                    f"mean log(R)={mean_log_R:.1f} | "
                    f"yield={mean_selected_yield:.3f} | "
                    f"capture={float(np.mean(np.array(batch_final.global_attrs.get('metric_capture_rate', [0])))):.2f}"
                )

    print(f"\nTrajectory CSV: {traj_path}")
    print(f"Summary CSV:    {summ_path}")
    return results, traj_path, summ_path


def main():
    """CLI entry point for basin stability experiment."""
    _run_cli()


def _run_cli():
    parser = argparse.ArgumentParser(description="Basin Stability Experiment")
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 seeds")
    parser.add_argument("--plot", action="store_true", help="Generate plots after sweep")
    parser.add_argument("--vmap", action="store_true",
                        help="Use vmap over seeds (batched GPU execution)")
    parser.add_argument("--tracking", type=str, default="both",
                        choices=["both", "predictive", "non_predictive"],
                        help="Which tracking modes to run")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output CSVs and plots")
    args = parser.parse_args()

    n_seeds = 10 if args.quick else args.n_seeds

    # Select tracking modes
    if args.tracking == "both":
        tracking_modes = TRACKING_MODES
    else:
        tracking_modes = {args.tracking: TRACKING_MODES[args.tracking]}

    print("=" * 60)
    print("Basin Stability Experiment")
    print("=" * 60)
    print(f"Agents: {args.n_agents}, Seeds: {n_seeds}")
    print(f"Mechanisms: PDD, PRD, PLD")
    print(f"Tracking modes: {list(tracking_modes.keys())}")
    print(f"Adversarial fractions: 0-60%")
    print(f"Metrics: {sorted(BASIN_METRICS.keys())}")
    print(f"Mode: {'vmap (batched)' if args.vmap else 'sequential'}")
    print(f"Devices: {jax.devices()}")
    print()

    sweep_fn = run_sweep_vmap if args.vmap else run_sweep
    results, traj_path, summ_path = sweep_fn(
        n_agents=args.n_agents,
        n_seeds=n_seeds,
        tracking_modes=tracking_modes,
        output_dir=args.output_dir,
    )

    # Summary tables
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print("\n" + "=" * 85)
    print("SURVIVAL RATE (fraction of seeds where R > R_min at T=200)")
    print("=" * 85)
    header = f"{'Condition':<25}"
    for frac in fracs:
        header += f" {frac:>7.0%}"
    print(header)
    print("-" * 85)
    for cond_key in sorted(results.keys()):
        row = f"{cond_key:<25}"
        for frac in fracs:
            data = results[cond_key].get(str(frac))
            if data:
                row += f" {data['survival_rate']:>7.0%}"
            else:
                row += f" {'---':>7}"
        print(row)

    print("\n" + "=" * 85)
    print("MEDIAN FINAL RESOURCE LEVEL")
    print("=" * 85)
    header = f"{'Condition':<25}"
    for frac in fracs:
        header += f" {frac:>9.0%}"
    print(header)
    print("-" * 85)
    for cond_key in sorted(results.keys()):
        row = f"{cond_key:<25}"
        for frac in fracs:
            data = results[cond_key].get(str(frac))
            if data:
                val = data['median_final_R']
                if val > 1e6:
                    row += f" {val:>9.1e}"
                else:
                    row += f" {val:>9.1f}"
            else:
                row += f" {'---':>9}"
        print(row)
    print("=" * 85)

    if args.plot:
        from experiments.basin_stability.plots import (
            plot_resources_vs_adversarial,
            plot_resource_trajectories,
        )
        plot_dir = Path(__file__).parent / "results"
        fig_a = plot_resources_vs_adversarial(
            summ_path, output_path=plot_dir / "resources_vs_adversarial.png"
        )
        for mech in ["pdd", "prd", "pld"]:
            fig_b = plot_resource_trajectories(
                traj_path, mechanism=mech,
                output_path=plot_dir / f"trajectories_{mech}.png"
            )
        print(f"\nPlots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
