"""CLI: run a benchmark scenario, print the scorecard, write it as a run record.

Each run leaves ``runs/<UTC stamp>_<scenario>/`` with ``manifest.json`` (env, config,
conditions and schedules, seeds, horizon, metric ids, engine) and ``scorecard.json``.
Records are small and belong in git.

    python -m experiments.benchmark.run                                # commons, full
    python -m experiments.benchmark.run --scenario compute_economy     # the economy
    python -m experiments.benchmark.run --smoke                        # tiny sanity run
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile

from cilib import runs
from cilib.environments import make_env

from cilib.lab.analysis.influence import influence_preserved

from .harness import run_benchmark
from .scenarios import SCENARIOS

RUNS_DIR = pathlib.Path(__file__).parent / "runs"


def print_scorecard(scorecard: dict) -> None:
    scenario = SCENARIOS[scorecard["scenario"]]
    meta = scorecard["meta"]
    print(f"\nscenario: {scorecard['scenario']}   "
          f"(n_seeds={meta['n_seeds']}, T={meta['T']}, seed={meta['seed']})\n")
    cols = scenario.headline_metrics
    header = (f"{'condition':<24}{'influence_preserved':>20}{'exercised':>11}"
              + "".join(f"{c[:18]:>20}" for c in cols))
    print(header)
    print("-" * len(header))
    for row in scorecard["rows"]:
        m = row["metrics"]
        print(f"{row['condition']:<24}{row['influence_preserved']:>20.3f}"
              f"{row['exercised_influence']:>11.3f}"
              + "".join(f"{m[c]['point']:>20.3f}" for c in cols))
    print("\ninfluence_preserved: share of baseline-lost CAUSAL influence recovered;"
          "\n  exercised = raw counterfactual responsiveness (paired same-key rollouts;"
          "\n  instrument per scenario - see experiments/benchmark/scenarios.py).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="governed_commons",
                        choices=sorted(SCENARIOS))
    parser.add_argument("--smoke", action="store_true", help="tiny run (2 seeds, 12 steps)")
    parser.add_argument("--n-seeds", type=int, default=32)
    parser.add_argument("--T", type=int, default=None,
                        help="horizon; defaults to the scenario's default_T")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_seeds, T = (2, 12) if args.smoke else (args.n_seeds, args.T)
    scorecard = run_benchmark(scenario=args.scenario, n_seeds=n_seeds, T=T,
                              seed=args.seed)
    print_scorecard(scorecard)

    scn = SCENARIOS[args.scenario]
    overrides = scorecard["meta"]["env_overrides"]
    env = make_env(scn.env_name, **overrides)
    # the two headline columns are instruments, not trace metrics; they get ids too
    metrics = {**env.metrics, "exercised_influence": scn.influence_fn,
               "influence_preserved": influence_preserved}
    m = runs.manifest(env=scn.env_name, config=env.config, overrides=overrides,
                      T=scorecard["meta"]["T"], seed0=args.seed, n_seeds=n_seeds,
                      kind="scorecard", conditions=scn.conditions, metrics=metrics,
                      notes=f"benchmark scenario {args.scenario}")
    root = pathlib.Path(tempfile.gettempdir()) / "benchmark_smoke" if args.smoke else RUNS_DIR
    out = runs.write(runs.run_dir(root, args.scenario, m["created"]), m,
                     files={"scorecard.json": scorecard})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
