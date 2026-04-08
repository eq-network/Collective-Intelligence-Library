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
                baseline_evals.append(float(np.mean(arr[eval_start:])))
            R_coop = np.mean(baseline_evals)
            print(f"  R_coop = {R_coop:.2f}")

            results[condition_key] = {"R_coop": float(R_coop)}

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

                # Compute basin stability
                eval_means = []
                for state in final_states:
                    arr = np.array(state.global_attrs["metric_resource_level"])
                    eval_start = int(len(arr) * 0.75)
                    eval_means.append(float(np.mean(arr[eval_start:])))

                n_stable = sum(1 for v in eval_means if v > R_coop)
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

            # Cooperative baseline (0 adversaries)
            n_baseline = min(50, n_seeds)
            key, subkey = jr.split(master_key)
            batch_final = run_batched(
                mechanism, n_agents, 0, subkey, n_baseline, T=T,
                metrics=metrics, tracking_lambda=tracking_lambda,
            )
            baseline_arrs = np.array(batch_final.global_attrs["metric_resource_level"])
            eval_start = int(T * 0.75)
            R_coop = float(np.mean(baseline_arrs[:, eval_start:]))
            print(f"  R_coop = {R_coop:.2f} (from {n_baseline} vmapped seeds)")

            results[condition_key] = {"R_coop": float(R_coop)}

            for adv_frac in adversarial_fractions:
                n_adversarial = int(n_agents * adv_frac)

                key, subkey = jr.split(key)
                batch_final = run_batched(
                    mechanism, n_agents, n_adversarial, subkey, n_seeds,
                    T=T, metrics=metrics, tracking_lambda=tracking_lambda,
                )

                # Extract metric arrays: shape (n_seeds, T)
                resource_arrs = np.array(batch_final.global_attrs["metric_resource_level"])
                eval_means = np.mean(resource_arrs[:, eval_start:], axis=1)

                n_stable = int(np.sum(eval_means > R_coop))
                n_total = n_seeds
                bs = n_stable / n_total
                ci_low, ci_high = wilson_ci(n_stable, n_total)

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
                summary_metrics = {}
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
                            "basin_stability", "bs_ci_lower", "bs_ci_upper",
                        ]
                        cols += sorted(summary_metrics.keys())
                        f.write(",".join(cols) + "\n")
                    vals = [mechanism, tracking_name, str(tracking_lambda),
                            str(adv_frac), str(n_seeds),
                            str(bs), str(ci_low), str(ci_high)]
                    vals += [str(summary_metrics[k]) for k in sorted(summary_metrics.keys())]
                    f.write(",".join(vals) + "\n")

                results[condition_key][str(adv_frac)] = {
                    "basin_stability": bs,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "n_stable": n_stable,
                    "n_total": n_total,
                }

                print(
                    f"  adv={adv_frac:.0%} -> BS={bs:.3f} "
                    f"[{ci_low:.3f}, {ci_high:.3f}] ({n_seeds} seeds vmapped)"
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

    # Summary table
    print("\n" + "=" * 80)
    print("BASIN STABILITY")
    print("=" * 80)
    header = f"{'Condition':<25}"
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        header += f" {frac:>7.0%}"
    print(header)
    print("-" * 80)
    for cond_key in sorted(results.keys()):
        row = f"{cond_key:<25}"
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            data = results[cond_key].get(str(frac))
            if data:
                row += f" {data['basin_stability']:>7.3f}"
            else:
                row += f" {'---':>7}"
        print(row)
    print("=" * 80)

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
