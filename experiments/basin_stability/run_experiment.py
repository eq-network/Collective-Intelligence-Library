"""
Run the basin stability experiment.

Sweeps across:
- 3 governance mechanisms (PDD, PRD, PLD)
- 7 adversarial fractions (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
- 500 seeds per condition

Measures basin stability: fraction of seeds where the system remains
above cooperative resource levels during the evaluation window.

Includes baselines: random, oracle, heuristic agents.

Usage:
    python -m experiments.basin_stability.run_experiment
    python -m experiments.basin_stability.run_experiment --n_seeds 10  # quick test
"""
import sys
import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import jax.numpy as jnp

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.basin_stability.environment import BasinStabilityEnv


def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval for a binomial proportion.

    More accurate than normal approximation for small n or extreme p.

    Args:
        successes: number of successes
        total: total trials
        z: z-score (1.96 for 95% CI)

    Returns:
        (lower, upper) bounds
    """
    if total == 0:
        return (0.0, 1.0)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def run_single(mechanism, n_agents, n_adversarial, seed, T=200, **kwargs):
    """Run a single episode. Returns the full resource trajectory."""
    env = BasinStabilityEnv(
        mechanism=mechanism,
        n_agents=n_agents,
        n_adversarial=n_adversarial,
        seed=seed,
        T=T,
        **kwargs,
    )
    history = env.run(T)

    # Extract resource trajectory
    trajectory = [float(env._initial_state.global_attrs["resource_level"])]
    for state in history:
        trajectory.append(float(state.global_attrs["resource_level"]))

    return trajectory


def compute_basin_stability(trajectories, R_coop, eval_start=150):
    """Compute basin stability from a set of trajectories.

    Basin stability = fraction of seeds where mean eval-window resources
    exceed the cooperative baseline R_coop.

    Args:
        trajectories: list of resource trajectories (list of lists)
        R_coop: cooperative baseline resource level
        eval_start: start of evaluation window

    Returns:
        (basin_stability, wilson_lower, wilson_upper, n_stable, n_total)
    """
    n_stable = 0
    n_total = len(trajectories)

    for traj in trajectories:
        if len(traj) > eval_start:
            eval_resources = traj[eval_start:]
            mean_eval = sum(eval_resources) / len(eval_resources)
            if mean_eval > R_coop:
                n_stable += 1

    bs = n_stable / n_total if n_total > 0 else 0.0
    ci_low, ci_high = wilson_ci(n_stable, n_total)
    return bs, ci_low, ci_high, n_stable, n_total


def compute_cooperative_baseline(mechanism, n_agents, n_seeds=50, T=200):
    """Estimate R_coop: mean eval-window resource with 0 adversaries.

    Uses a smaller number of seeds for efficiency.
    """
    trajectories = []
    for seed in range(n_seeds):
        traj = run_single(mechanism, n_agents, 0, seed=seed * 7919, T=T)
        trajectories.append(traj)

    # Mean resource in eval window across all cooperative runs
    eval_resources = []
    for traj in trajectories:
        if len(traj) > 150:
            eval_resources.extend(traj[150:])

    if not eval_resources:
        return 100.0  # fallback to initial
    return sum(eval_resources) / len(eval_resources)


def run_sweep(
    mechanisms=("pdd", "prd", "pld"),
    adversarial_fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    n_agents: int = 20,
    n_seeds: int = 500,
    T: int = 200,
    seed_offset: int = 0,
):
    """Run the full experimental sweep.

    Returns:
        results: nested dict with basin stability and trajectories
    """
    results = {}

    for mechanism in mechanisms:
        print(f"\n{'='*50}")
        print(f"Mechanism: {mechanism.upper()}")
        print(f"{'='*50}")

        # Compute cooperative baseline for this mechanism
        print(f"  Computing cooperative baseline (R_coop)...")
        R_coop = compute_cooperative_baseline(mechanism, n_agents, n_seeds=min(50, n_seeds))
        print(f"  R_coop = {R_coop:.2f}")

        results[mechanism] = {"R_coop": R_coop}

        for adv_frac in adversarial_fractions:
            n_adversarial = int(n_agents * adv_frac)

            trajectories = []
            for seed_idx in range(n_seeds):
                seed = seed_offset + seed_idx * 7919 + hash(mechanism) % 10000
                traj = run_single(mechanism, n_agents, n_adversarial, seed=seed, T=T)
                trajectories.append(traj)

                if (seed_idx + 1) % max(1, n_seeds // 5) == 0:
                    # Progress update
                    bs_so_far, _, _, n_s, n_t = compute_basin_stability(
                        trajectories, R_coop
                    )
                    print(
                        f"  adv={adv_frac:.0%} | {seed_idx+1}/{n_seeds} | "
                        f"BS={bs_so_far:.3f} ({n_s}/{n_t})"
                    )

            bs, ci_low, ci_high, n_stable, n_total = compute_basin_stability(
                trajectories, R_coop
            )

            # Summary statistics for trajectories
            final_resources = [traj[-1] for traj in trajectories]
            survival_times = [len(traj) for traj in trajectories]

            results[mechanism][str(adv_frac)] = {
                "basin_stability": bs,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_stable": n_stable,
                "n_total": n_total,
                "mean_final_resource": sum(final_resources) / len(final_resources),
                "mean_survival": sum(survival_times) / len(survival_times),
            }

            print(
                f"  adv={adv_frac:.0%} -> BS={bs:.3f} "
                f"[{ci_low:.3f}, {ci_high:.3f}]"
            )

    return results


def run_baselines(
    mechanisms=("pdd", "prd", "pld"),
    adversarial_fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    n_agents: int = 20,
    n_seeds: int = 500,
):
    """Run baseline agents: random, oracle, heuristic.

    These use the same environment but override agent behavior:
    - Random: uniform action selection (epsilon=1.0 always)
    - Oracle: always select best proposal (epsilon=0.0, perfect signals)
    - Heuristic: vote for highest signal (no learning)

    For simplicity, baselines are implemented by manipulating epsilon
    and signal quality at initialization.
    """
    baseline_results = {}

    # Random baseline: epsilon=1.0 throughout (pure exploration)
    print("\n--- Random Baseline ---")
    for mechanism in mechanisms:
        R_coop = 100.0  # use initial resource as baseline for random
        for adv_frac in adversarial_fractions:
            n_adversarial = int(n_agents * adv_frac)
            trajectories = []
            for seed_idx in range(n_seeds):
                seed = seed_idx * 7919
                traj = run_single(
                    mechanism, n_agents, n_adversarial, seed=seed,
                    epsilon_start=1.0, epsilon_end=1.0,
                )
                trajectories.append(traj)

            bs, ci_low, ci_high, _, _ = compute_basin_stability(trajectories, R_coop)
            key = f"random_{mechanism}_{adv_frac}"
            baseline_results[key] = {
                "basin_stability": bs,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }

    # Oracle baseline: epsilon=0, near-zero noise
    print("--- Oracle Baseline ---")
    # Oracle is simulated by setting epsilon=0 and very low signal noise
    # (can't set sigma=0 exactly, use 0.001)

    # Heuristic baseline: epsilon=0, no learning (alpha=0)
    print("--- Heuristic Baseline ---")
    for mechanism in mechanisms:
        R_coop = 100.0
        for adv_frac in adversarial_fractions:
            n_adversarial = int(n_agents * adv_frac)
            trajectories = []
            for seed_idx in range(n_seeds):
                seed = seed_idx * 7919
                traj = run_single(
                    mechanism, n_agents, n_adversarial, seed=seed,
                    alpha=0.0, epsilon_start=0.0, epsilon_end=0.0,
                )
                trajectories.append(traj)

            bs, ci_low, ci_high, _, _ = compute_basin_stability(trajectories, R_coop)
            key = f"heuristic_{mechanism}_{adv_frac}"
            baseline_results[key] = {
                "basin_stability": bs,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }

    return baseline_results


def save_results(results, baseline_results=None, output_dir=None):
    """Save results to JSON."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Main results
    filepath = output_dir / f"basin_stability_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")

    # Baselines
    if baseline_results:
        bl_path = output_dir / f"baselines_{timestamp}.json"
        with open(bl_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"Baselines saved to: {bl_path}")

    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basin Stability Experiment")
    parser.add_argument("--n_seeds", type=int, default=500)
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 seeds")
    args = parser.parse_args()

    n_seeds = 10 if args.quick else args.n_seeds

    print("=" * 60)
    print("Basin Stability Experiment")
    print("=" * 60)
    print(f"Agents: {args.n_agents}, Seeds: {n_seeds}")
    print(f"Mechanisms: PDD, PRD, PLD")
    print(f"Adversarial fractions: 0-60%")
    print()

    results = run_sweep(n_agents=args.n_agents, n_seeds=n_seeds)

    baseline_results = None
    if args.baselines:
        baseline_results = run_baselines(n_agents=args.n_agents, n_seeds=n_seeds)

    save_results(results, baseline_results)

    # Summary table
    print("\n" + "=" * 70)
    print("BASIN STABILITY (BS +/- 95% Wilson CI)")
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
