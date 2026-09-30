"""Export one rollout as web-playground JSON (docs/web-trajectory-contract.md).

The lab page's playground consumes a flat trajectory schema that any producer —
the in-browser JS port, a future JAX endpoint, or this script — emits
identically. This is the file-producer: it runs a registered environment once
and maps the trace onto the contract mechanically (no per-environment code):

    (T,) fields -> "global";  (T, N) varying -> "node";  (T, N) constant -> "static"

Used for parity fixtures against the browser port and, later, as StaticEngine
payloads for published-run replays. Not a study (no sweep, no CIs) — for that,
see experiments/benchmark/.

The payload lands as ``trajectory.json`` inside a run record (``cilib.runs``):
a directory with a ``manifest.json`` carrying the address (env, config,
mechanisms and schedules, T, seed), metric ids, engine and checksums.

    python examples/05_export_trajectory.py --env governed_commons --T 500
    python examples/05_export_trajectory.py --env compute_economy \
        --condition tax_and_ownership_cap --T 500 --out runs/econ
"""
import argparse
import dataclasses
import os
import tempfile

import jax.random as jr

from cilib import runs
from cilib.core.schedule import ScheduleSpec, apply_schedule
from cilib.environments import (
    compute_economy, coupled_society, governed_commons, influence_exchange,
    make_env, value_contagion,
)
from cilib.environments.system_graph import system_graph
from cilib.environments.webexport import trajectory_payload
from cilib.mechanisms import (
    REGISTRY as MECHANISMS,
    AIRevenueTaxConfig, EnforcedAITaxConfig, InfluenceCapConfig,
    OwnershipCapConfig, QuotaVoteConfig, SanctionConfig, SortitionConfig,
)

# Kept in sync by hand with experiments/benchmark/scenarios.py (same convention
# as examples/04): this script depends only on the installed package.
# A condition is either a list of (mechanism, config, schedule) triples or a
# dict {"mechanisms": [...], "overrides": {...}} — value_contagion's corners
# are config dials, not mechanisms.
CONDITIONS = {
    "governed_commons": {
        "baseline": [],
        "quota_voting": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5))],
        "graduated_sanctions": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5)),
                                ("graduated_sanction", SanctionConfig(), None)],
    },
    "compute_economy": {
        "baseline": [],
        "ai_revenue_tax": [("ai_revenue_tax", AIRevenueTaxConfig(tax_rate=0.5),
                            ScheduleSpec(onset=50))],
        "tax_and_ownership_cap": [("ai_revenue_tax", AIRevenueTaxConfig(tax_rate=0.5),
                                   ScheduleSpec(onset=50)),
                                  ("ownership_cap", OwnershipCapConfig(cap_share=0.35), None)],
    },
    "value_contagion": {
        "pluralism": {"overrides": {"ai_homophily": 0.05, "p_advantage": 1.0}},
        "assimilation": {"overrides": {"ai_homophily": 0.05, "p_advantage": 6.0}},
        "parallel_cultures": {"overrides": {"ai_homophily": 0.9, "p_advantage": 1.0}},
        "displacement": {"overrides": {"ai_homophily": 0.9, "p_advantage": 6.0}},
    },
    "influence_exchange": {
        "organic": {"overrides": {"amp_onset": 10_000}},
        "amplified": [],
        "sortition_only": [("sortition", SortitionConfig(), ScheduleSpec(cadence=15))],
        "defended": [("sortition", SortitionConfig(), ScheduleSpec(cadence=15)),
                     ("influence_cap", InfluenceCapConfig(), None)],
    },
    "coupled_society": {
        "sealed_undefended": {"overrides": {"kappa": 0.0}},
        "coupled_undefended": [],
        "coupled_defended": [("enforced_ai_tax", EnforcedAITaxConfig(), ScheduleSpec(onset=50)),
                             ("sortition", SortitionConfig(), ScheduleSpec(cadence=15)),
                             ("influence_cap", InfluenceCapConfig(), None)],
        "sealed_defended": {
            "mechanisms": [("enforced_ai_tax", EnforcedAITaxConfig(), ScheduleSpec(onset=50)),
                           ("sortition", SortitionConfig(), ScheduleSpec(cadence=15)),
                           ("influence_cap", InfluenceCapConfig(), None)],
            "overrides": {"kappa": 0.0},
        },
    },
}

# env -> its module, for the derived system graph (build_steps/make_state are
# the uniform subpackage convention; the graph is pipeline metadata, not data).
ENV_MODULES = {
    "governed_commons": governed_commons,
    "compute_economy": compute_economy,
    "value_contagion": value_contagion,
    "influence_exchange": influence_exchange,
    "coupled_society": coupled_society,
}


def export(env_name: str, condition: str, n_steps: int, seed: int) -> tuple:
    """One rollout as ``(manifest, payload)``: the record's address and the contract JSON."""
    cond = CONDITIONS[env_name][condition]
    triples, overrides = (cond.get("mechanisms", []), cond.get("overrides", {})) \
        if isinstance(cond, dict) else (cond, {})
    mechs = tuple(apply_schedule(MECHANISMS[key](cfg), sched)
                  for key, cfg, sched in triples)
    env = make_env(env_name, mechanisms=mechs, **overrides)
    finals, trace = env.run(jr.PRNGKey(seed), n_steps)

    mod = ENV_MODULES.get(env_name)
    system = None
    if mod is not None:          # the pipeline DAG read as communication
        system = system_graph(mod.build_steps(env.config, mechs),
                              mod.make_state(env.config, jr.PRNGKey(seed)))

    # the generic mapping lives in cilib.environments.webexport (shared with the
    # sweep-bundle exporter); defaults here reproduce the historical output
    payload = trajectory_payload(
        trace, finals, game_id=env_name, n_steps=n_steps, seed=seed,
        params={**dataclasses.asdict(env.config), "condition": condition},
        scalars=env.evaluate(trace), system=system)
    manifest = runs.manifest(env=env_name, config=env.config, T=n_steps, seed0=seed,
                             n_seeds=1, kind="trajectory", mechanisms=triples,
                             overrides=overrides, metrics=env.metrics,
                             notes=f"condition {condition}")
    return manifest, payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="governed_commons", choices=sorted(CONDITIONS))
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None,
                        help="record directory (default: examples/runs/<stamp>_<env>_<condition>_s<seed>)")
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    args = parser.parse_args()

    if args.condition not in CONDITIONS[args.env]:
        parser.error(f"--condition must be one of {sorted(CONDITIONS[args.env])} for {args.env}")
    if args.smoke:
        args.T = 5
        # A fresh dir per run: a record dir is write-once, so a fixed path fails the second time.
        args.out = args.out or os.path.join(tempfile.mkdtemp(prefix="cilib_export_smoke_"), "record")

    manifest, payload = export(args.env, args.condition, args.T, args.seed)
    out = args.out or runs.run_dir(os.path.join(os.path.dirname(__file__), "runs"),
                                   f"{args.env}_{args.condition}_s{args.seed}",
                                   manifest["created"])
    runs.write(out, manifest, files={"trajectory.json": payload})
    scalars = "  ".join(f"{k}={v:.3f}" for k, v in payload["meta"]["scalars"].items())
    print(f"wrote {out}  (T={args.T}, N={payload['meta']['N']})\n{scalars}")
