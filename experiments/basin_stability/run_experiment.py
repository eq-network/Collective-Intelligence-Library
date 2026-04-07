"""
Run the basin stability experiment.

Sweeps across:
- 3 governance mechanisms (PDD, PRD, PLD)
- 7 adversarial fractions (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
- 500 seeds per condition

Uses inline metric arrays filled via lax.scan for JIT-compiled execution.
Outputs CSV files for analysis and plotting.

Usage:
    python -m experiments.basin_stability.run_experiment --quick
    python -m experiments.basin_stability.run_experiment --n_seeds 500
    python -m experiments.basin_stability.run_experiment --n_seeds 500 --plot
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
from metrics.export import write_trajectory_csv, write_summary_csv, extract_metric_arrays


# Default metrics for this experiment: economic + governance
BASIN_METRICS = {**ECONOMIC_METRICS, **GOVERNANCE_METRICS}


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
               metrics=None, **kwargs):
    """Run a single episode via env.run_scan(). Returns final GraphState with filled metric arrays."""
    if metrics is None:
        metrics = BASIN_METRICS

    env = BasinStabilityEnv(
        mechanism=mechanism,
        n_agents=n_agents,
        n_adversarial=n_adversarial,
        seed=seed,
        T=T,
        metrics=metrics,
        **kwargs,
    )
    return env.run(T)


def compute_basin_stability(final_states, metric_name="resource_level", eval_frac=0.75):
    """Compute per-seed eval-window means from final states with filled metric arrays.

    Args:
        final_states: list of final GraphStates with filled metric arrays
        metric_name: which metric to evaluate
        eval_frac: fraction of timesteps to use as eval window start (default 0.75 = last 25%)

    Returns:
        list of mean metric values in the eval window, one per seed
    """
    eval_values = []
    for state in final_states:
        arr = np.array(state.global_attrs[f"metric_{metric_name}"])
        eval_start = int(len(arr) * eval_frac)
        eval_values.append(float(np.mean(arr[eval_start:])))
    return eval_values


def run_sweep(
    mechanisms=("pdd", "prd", "pld"),
    adversarial_fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    n_agents: int = 20,
    n_seeds: int = 500,
    T: int = 200,
    output_dir: str = None,
    metrics: dict = None,
):
    """Run the full experimental sweep with CSV output.

    Streams trajectory CSV rows as each seed completes.
    Writes summary CSV after all seeds per condition.
    """
    if metrics is None:
        metrics = BASIN_METRICS
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
        print(f"\n{'='*50}")
        print(f"Mechanism: {mechanism.upper()}")
        print(f"{'='*50}")

        # Compute cooperative baseline (0 adversaries)
        print("  Computing cooperative baseline (R_coop)...")
        n_baseline = min(50, n_seeds)
        baseline_evals = []
        for s in range(n_baseline):
            final = run_single(mechanism, n_agents, 0, seed=s * 7919, T=T, metrics=metrics)
            arr = np.array(final.global_attrs["metric_resource_level"])
            eval_start = int(len(arr) * 0.75)
            baseline_evals.append(float(np.mean(arr[eval_start:])))
        R_coop = np.mean(baseline_evals)
        print(f"  R_coop = {R_coop:.2f}")

        results[mechanism] = {"R_coop": float(R_coop)}

        for adv_frac in adversarial_fractions:
            n_adversarial = int(n_agents * adv_frac)

            final_states = []
            for seed_idx in range(n_seeds):
                seed = seed_idx * 7919 + hash(mechanism) % 10000

                final = run_single(
                    mechanism, n_agents, n_adversarial,
                    seed=seed, T=T, metrics=metrics,
                )
                final_states.append(final)

                # Stream trajectory CSV
                run_meta = {
                    "adversarial_fraction": adv_frac,
                    "mechanism": mechanism,
                    "seed": seed,
                }
                write_trajectory_csv(traj_path, final, metric_names, run_meta)

                # Progress
                if (seed_idx + 1) % max(1, n_seeds // 5) == 0:
                    evals = compute_basin_stability(final_states)
                    n_stable = sum(1 for v in evals if v > R_coop)
                    bs = n_stable / len(evals)
                    print(
                        f"  adv={adv_frac:.0%} | {seed_idx+1}/{n_seeds} | "
                        f"BS={bs:.3f} ({n_stable}/{len(evals)})"
                    )

            # Compute basin stability for this condition
            eval_means = compute_basin_stability(final_states)
            n_stable = sum(1 for v in eval_means if v > R_coop)
            n_total = len(eval_means)
            bs = n_stable / n_total
            ci_low, ci_high = wilson_ci(n_stable, n_total)

            # Write summary CSV row
            extra_cols = {
                "basin_stability": bs,
                "bs_ci_lower": ci_low,
                "bs_ci_upper": ci_high,
            }
            condition_meta = {
                "adversarial_fraction": adv_frac,
                "mechanism": mechanism,
            }
            write_summary_csv(summ_path, condition_meta, final_states, metric_names, extra_cols)

            results[mechanism][str(adv_frac)] = {
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
    adversarial_fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    n_agents: int = 20,
    n_seeds: int = 500,
    T: int = 200,
    output_dir: str = None,
    metrics: dict = None,
    master_seed: int = 0,
):
    """Run the full sweep with vmap over seeds — one GPU kernel per condition.

    Same output format as run_sweep (CSV files), but all seeds for a given
    (mechanism, adversarial_fraction) are batched into a single vmap call.
    """
    if metrics is None:
        metrics = BASIN_METRICS
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
        print(f"\n{'='*50}")
        print(f"Mechanism: {mechanism.upper()}")
        print(f"{'='*50}")

        # Cooperative baseline (0 adversaries)
        n_baseline = min(50, n_seeds)
        key, subkey = jr.split(master_key)
        batch_final = run_batched(
            mechanism, n_agents, 0, subkey, n_baseline, T=T, metrics=metrics,
        )
        baseline_arrs = np.array(batch_final.global_attrs["metric_resource_level"])  # (n_baseline, T)
        eval_start = int(T * 0.75)
        R_coop = float(np.mean(baseline_arrs[:, eval_start:]))
        print(f"  R_coop = {R_coop:.2f} (from {n_baseline} vmapped seeds)")

        results[mechanism] = {"R_coop": float(R_coop)}

        for adv_frac in adversarial_fractions:
            n_adversarial = int(n_agents * adv_frac)

            key, subkey = jr.split(key)
            batch_final = run_batched(
                mechanism, n_agents, n_adversarial, subkey, n_seeds,
                T=T, metrics=metrics,
            )

            # Extract metric arrays: shape (n_seeds, T)
            resource_arrs = np.array(batch_final.global_attrs["metric_resource_level"])
            eval_means = np.mean(resource_arrs[:, eval_start:], axis=1)  # (n_seeds,)

            n_stable = int(np.sum(eval_means > R_coop))
            n_total = n_seeds
            bs = n_stable / n_total
            ci_low, ci_high = wilson_ci(n_stable, n_total)

            # Write trajectory CSV (one row per seed per timestep)
            for seed_idx in range(n_seeds):
                run_meta = {
                    "adversarial_fraction": adv_frac,
                    "mechanism": mechanism,
                    "seed": seed_idx,
                }
                # Extract single-seed final state for CSV writer
                # We write directly from the batched arrays
                metric_row = {}
                for mn in metric_names:
                    arr = np.array(batch_final.global_attrs[f"metric_{mn}"])
                    metric_row[mn] = arr[seed_idx]
                # Write trajectory rows for this seed
                with open(traj_path, "a") as f:
                    if traj_path.stat().st_size == 0:
                        header = ",".join(["mechanism", "adversarial_fraction", "seed", "step"] + metric_names)
                        f.write(header + "\n")
                    for step in range(T):
                        vals = [str(metric_row[mn][step]) for mn in metric_names]
                        row = f"{mechanism},{adv_frac},{seed_idx},{step}," + ",".join(vals)
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
                    cols = ["mechanism", "adversarial_fraction", "n_seeds",
                            "basin_stability", "bs_ci_lower", "bs_ci_upper"]
                    cols += sorted(summary_metrics.keys())
                    f.write(",".join(cols) + "\n")
                vals = [mechanism, str(adv_frac), str(n_seeds),
                        str(bs), str(ci_low), str(ci_high)]
                vals += [str(summary_metrics[k]) for k in sorted(summary_metrics.keys())]
                f.write(",".join(vals) + "\n")

            results[mechanism][str(adv_frac)] = {
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
    parser.add_argument("--n_seeds", type=int, default=500)
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 seeds")
    parser.add_argument("--plot", action="store_true", help="Generate plots after sweep")
    parser.add_argument("--vmap", action="store_true",
                        help="Use vmap over seeds (batched GPU execution)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output CSVs and plots")
    args = parser.parse_args()

    n_seeds = 10 if args.quick else args.n_seeds

    print("=" * 60)
    print("Basin Stability Experiment")
    print("=" * 60)
    print(f"Agents: {args.n_agents}, Seeds: {n_seeds}")
    print(f"Mechanisms: PDD, PRD, PLD")
    print(f"Adversarial fractions: 0-60%")
    print(f"Metrics: {sorted(BASIN_METRICS.keys())}")
    print(f"Mode: {'vmap (batched)' if args.vmap else 'sequential'}")
    print(f"Devices: {jax.devices()}")
    print()

    sweep_fn = run_sweep_vmap if args.vmap else run_sweep
    results, traj_path, summ_path = sweep_fn(
        n_agents=args.n_agents,
        n_seeds=n_seeds,
        output_dir=args.output_dir,
    )

    # Summary table
    print("\n" + "=" * 70)
    print("BASIN STABILITY")
    print("=" * 70)
    header = f"{'Mech':<8}"
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        header += f" {frac:>7.0%}"
    print(header)
    print("-" * 70)
    for mech in ["pdd", "prd", "pld"]:
        row = f"{mech.upper():<8}"
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            data = results[mech][str(frac)]
            row += f" {data['basin_stability']:>7.3f}"
        print(row)
    print("=" * 70)

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
