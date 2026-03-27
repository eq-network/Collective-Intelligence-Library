"""
Run the governed commons harvest experiment.

Sweeps across:
- 4 governance mechanisms (none, PDD, PRD, PLD)
- 4 adversarial concentrations (0%, 25%, 50%, 75%)
- N replications per condition

Outputs phase diagram data: survival time as a function of
mechanism x adversarial concentration.

Usage:
    python -m experiments.governed_harvest.run_experiment
"""
import sys
import os
import jax.numpy as jnp
import json
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.governed_harvest.environment import GovernedHarvestEnv


def run_sweep(
    n_agents: int = 100,
    n_replications: int = 30,
    adversarial_fractions: tuple = (0.0, 0.25, 0.5, 0.75),
    mechanisms: tuple = ("none", "pdd", "prd", "pld"),
    seed: int = 42,
):
    """Run the full experimental sweep.

    Returns a nested dict:
        results[mechanism][adversarial_frac] = {
            "survival_times": [...],
            "mean_survival": float,
            ...
        }
    """
    results = {}

    for mechanism in mechanisms:
        results[mechanism] = {}
        for adv_frac in adversarial_fractions:
            n_adversarial = int(n_agents * adv_frac)

            survival_times = []
            final_resources = []

            for rep in range(n_replications):
                rep_seed = seed + rep * 1000 + hash(mechanism) % 10000

                env = GovernedHarvestEnv(
                    mechanism=mechanism,
                    n_agents=n_agents,
                    n_adversarial=n_adversarial,
                    seed=rep_seed,
                )
                history = env.run(200)

                survival_time = len(history)
                final_resource = float(env.state.global_attrs["resource_level"])
                survival_times.append(survival_time)
                final_resources.append(final_resource)

                if (rep + 1) % 10 == 0:
                    print(
                        f"  {mechanism} | adv={adv_frac:.0%} | "
                        f"rep {rep+1}/{n_replications} | "
                        f"mean survival: {sum(survival_times)/len(survival_times):.0f}"
                    )

            mean_surv = sum(survival_times) / len(survival_times)
            std_surv = float(jnp.std(jnp.array(survival_times)))

            results[mechanism][str(adv_frac)] = {
                "survival_times": survival_times,
                "mean_survival": mean_surv,
                "std_survival": std_surv,
                "final_resources": final_resources,
                "mean_final_resource": sum(final_resources) / len(final_resources),
            }

            print(
                f"  {mechanism} | adv={adv_frac:.0%} -> "
                f"survival: {mean_surv:.0f} +/- {std_surv:.1f}"
            )

    return results


def save_results(results: dict, output_dir: str = None):
    """Save results to JSON for later analysis/plotting."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"sweep_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    print("=" * 60)
    print("Governed Commons Harvest Experiment")
    print("=" * 60)
    print()
    print("Mechanisms: none, PDD, PRD, PLD")
    print(f"Agents: 100 (harvest levels [0-5], K=5000, r=0.4)")
    print("Adversarial fractions: 0%, 25%, 50%, 75%")
    print("Replications: 30 per condition")
    print()

    results = run_sweep(
        n_agents=100,
        n_replications=30,
        seed=42,
    )

    filepath = save_results(results)

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS: Mean Survival Time (steps out of 200)")
    print("=" * 60)
    print(f"{'Mechanism':<12} {'0%':>8} {'25%':>8} {'50%':>8} {'75%':>8}")
    print("-" * 44)
    for mech in ["none", "pdd", "prd", "pld"]:
        row = f"{mech.upper():<12}"
        for frac in ["0.0", "0.25", "0.5", "0.75"]:
            mean = results[mech][frac]["mean_survival"]
            row += f"{mean:>8.0f}"
        print(row)
    print("=" * 60)
