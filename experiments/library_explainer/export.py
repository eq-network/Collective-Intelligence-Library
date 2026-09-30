"""Write the /library/prototype explainer fixtures: manifest + pipeline-subsets
+ system-graphs + runs/.

    python -m experiments.library_explainer.export [--smoke] [--out DIR]

The explainer page on eq-network renders the library's own machinery from
versioned artifacts, per the CLAUDE.md boundary: execution batches come from
``compile_pipeline``'s own ordering (a 2^4 lookup over the graduated_sanctions
pipeline), pipeline views from ``system_graph()``, and every charted series
from a contract-v1.1 payload — nothing on the page steps state or re-derives
ordering. Paste destination: eq-network/apps/site/src/data/library-explainer/
(pasted, never hand-edited).

Every file is validated on write with Python ``jsonschema`` against
schema/explainer.schema.json; eq-network CI re-validates the same artifacts
with ajv — two independent validators of one shared schema are the drift
guard, the same pattern as experiments/gd_bundles.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import tempfile

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cilib.core.category import transform
from cilib.core.graph import GraphState
from cilib.core.pipeline import get_execution_order
from cilib.core.schedule import apply_schedule, scheduled
from cilib.environments import governed_commons, make_env, value_contagion
from cilib.environments.counterfactual import (
    collective_influence, intervention_response,
)
from cilib.environments.governed_commons import (
    per_capita_harvest, shift_preferences,
)
from cilib.environments.system_graph import system_graph
from cilib.environments.webexport import trajectory_payload
from cilib import runs
from cilib.mechanisms import REGISTRY as MECHANISMS
from cilib.metrics.families.spectral import fiedler_partition_alignment_of

from .config import (
    CONDITIONS, CURVE_CONDITIONS, CURVE_DELTA, CURVE_SEEDS, CURVE_T0S,
    CURVE_WINDOW, ENV, MAX_FILE_BYTES, MAX_TOTAL_BYTES, ROUND_DECIMALS,
    RUN_WHITELIST, SCHEDULE_COMBOS, SCHEDULE_TICKS, SCORECARD_CAVEAT,
    SCORECARD_DELTA, SCORECARD_DESCRIPTIVE, SCORECARD_SEEDS, SEED0,
    SUBSET_MECHANISMS, T,
)
from .snippets import extract as extract_snippets

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema",
                           "explainer.schema.json")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "dist-fixtures")


def _validators():
    try:
        import jsonschema
    except ImportError as e:                       # pragma: no cover
        raise SystemExit("fixture validation needs jsonschema: "
                         "pip install -e .[export]") from e
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    def against(def_name):
        return jsonschema.Draft202012Validator(
            {"$ref": f"#/$defs/{def_name}", "$defs": schema["$defs"]})
    return {k: against(k) for k in (
        "manifest", "subsets", "systems", "run", "snippets", "stateShapes",
        "scheduleGolden", "scorecard", "influenceCurve", "graphMatrix")}


def _mechanisms_for(triples):
    return tuple(apply_schedule(MECHANISMS[key](cfg), sched)
                 for key, cfg, sched in triples)


# --- pipeline-subsets.json ------------------------------------------------------

def subset_transforms():
    """The BatchBoard lattice's transforms in program order, with display
    metadata. Steps are the real graduated_sanctions pipeline minus the
    trailing step_counter (bookkeeping)."""
    cfg = make_env(ENV).config
    mechs = _mechanisms_for(SUBSET_MECHANISMS)
    steps = governed_commons.build_steps(cfg, mechs)
    assert getattr(steps[-1], "name", "") == "step_counter", \
        "governed_commons pipeline no longer ends in step_counter"
    steps = steps[:-1]

    n_sub = len(steps) - len(mechs)
    names = ([getattr(t, "name") for t in steps[:n_sub]]
             + [key for key, _, _ in SUBSET_MECHANISMS])
    schedules = ([None] * n_sub
                 + [dataclasses.asdict(sched) if sched is not None else None
                    for _, _, sched in SUBSET_MECHANISMS])
    entries = [{"name": name,
                "reads": sorted(t.reads), "writes": sorted(t.writes),
                "schedule": sched}
               for name, sched, t in zip(names, schedules, steps)]
    return steps, entries


def _hazard_edges(steps, enabled):
    """[i, j, kinds] for enabled pairs, same set algebra and program-order
    orientation as core/pipeline.py (RAW / WAR / WAW)."""
    edges = []
    for a, i in enumerate(enabled):
        ri, wi = steps[i].reads, steps[i].writes
        for j in enabled[a + 1:]:
            rj, wj = steps[j].reads, steps[j].writes
            kinds = [k for k, shared in
                     (("RAW", wi & rj), ("WAR", ri & wj), ("WAW", wi & wj))
                     if shared]
            if kinds:
                edges.append([i, j, kinds])
    return edges


def build_subsets_fixture() -> dict:
    steps, entries = subset_transforms()
    n = len(steps)
    rows = []
    for mask in range(2 ** n):
        enabled = [i for i in range(n) if mask >> i & 1]
        sub = [steps[i] for i in enabled]
        order = get_execution_order(sub) if sub else []
        pos = {id(t): i for i, t in zip(enabled, sub)}
        batches = [[pos[id(t)] for t in batch] for batch in order]
        edges = _hazard_edges(steps, enabled)

        depth = {i: b for b, batch in enumerate(batches) for i in batch}
        assert all(depth[i] < depth[j] for i, j, _ in edges), \
            f"subset {enabled}: an edge does not run forward across batches"
        rows.append({"enabled": enabled, "edges": edges, "batches": batches})
    return {"env": ENV, "transforms": entries, "rows": rows}


# --- system-graphs.json ---------------------------------------------------------

def build_systems_fixture() -> dict:
    conditions = {}
    for name, triples in CONDITIONS.items():
        mechs = _mechanisms_for(triples)
        env = make_env(ENV, mechanisms=mechs)
        conditions[name] = system_graph(
            governed_commons.build_steps(env.config, mechs),
            governed_commons.make_state(env.config, jr.PRNGKey(SEED0)))
    return {"env": ENV, "conditions": conditions}


# --- runs/ ----------------------------------------------------------------------

def build_run_payload(condition: str, n_steps: int) -> dict:
    mechs = _mechanisms_for(CONDITIONS[condition])
    env = make_env(ENV, mechanisms=mechs)
    finals, trace = env.run(jr.PRNGKey(SEED0), n_steps)
    return trajectory_payload(
        trace, finals, game_id=ENV, n_steps=n_steps, seed=SEED0,
        params={**dataclasses.asdict(env.config), "condition": condition},
        scalars=env.evaluate(trace),
        whitelist=RUN_WHITELIST, round_decimals=ROUND_DECIMALS,
        include_adj=False)


# --- schedule-golden.json -------------------------------------------------------

def build_schedule_golden() -> dict:
    """Probe the engine's own scheduled() wrapper tick by tick — the page
    widget's predicate is pinned to these windows, not trusted."""
    @transform(reads=["fired"], writes=["fired"])
    def probe(state):
        return state.update_global_attr("fired", jnp.asarray(1.0))

    combos = []
    for cadence, phase_offset, onset in SCHEDULE_COMBOS:
        wrapped = scheduled(probe, cadence=cadence, phase_offset=phase_offset,
                            onset=onset)
        fires = []
        for t in range(SCHEDULE_TICKS):
            state = GraphState(
                node_types=jnp.zeros(1, dtype=jnp.int32), node_attrs={},
                adj_matrices={},
                global_attrs={"step": jnp.asarray(t), "fired": jnp.asarray(0.0)})
            fires.append(bool(wrapped(state).global_attrs["fired"] > 0))
        combos.append({"cadence": cadence, "phase_offset": phase_offset,
                       "onset": onset, "fires": fires})
    return {"ticks": SCHEDULE_TICKS, "combos": combos}


# --- state-shapes.json ----------------------------------------------------------

def build_state_shapes() -> dict:
    env = make_env(ENV)
    state = governed_commons.make_state(env.config, jr.PRNGKey(SEED0))

    def describe(mapping):
        return {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                for name, arr in mapping.items() if hasattr(arr, "shape")}

    return {
        "env": ENV,
        "N": int(state.node_types.shape[0]),
        "node_types_dtype": str(state.node_types.dtype),
        "fields": {
            "node_attrs": describe(state.node_attrs),
            "adj_matrices": describe(state.adj_matrices),
            "edge_attrs": describe(state.edge_attrs),
            "global_attrs": describe(state.global_attrs),
        },
    }


# --- scorecard.json -------------------------------------------------------------

def _stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": round(float(values.mean()), 4), "se": round(se, 4)}


def build_scorecard(n_steps: int, n_seeds: int) -> dict:
    """The benchmark's scenario-1 reading rebuilt at export time so provenance
    and caveat class travel inside the fixture. Causal column: paired same-key
    rollouts (collective_influence); descriptive columns: env.evaluate."""
    rows = []
    for name in CONDITIONS:
        mechs = _mechanisms_for(CONDITIONS[name])
        env = make_env(ENV, mechanisms=mechs)
        _, traces = env.run_batch(jr.PRNGKey(SEED0), n_seeds, n_steps)
        per_seed = jax.vmap(env.evaluate)(traces)
        influence = collective_influence(
            env, jr.PRNGKey(SEED0), n_seeds, n_steps, delta=SCORECARD_DELTA,
            perturb_fn=shift_preferences, outcome_fn=per_capita_harvest)
        values = {"influence_preserved": _stats(influence)}
        for metric_id in SCORECARD_DESCRIPTIVE:
            values[metric_id] = _stats(per_seed[metric_id])
        rows.append({"condition": name, "values": values})

    return {
        "env": ENV, "T": n_steps, "n_seeds": n_seeds, "seed0": SEED0,
        "caveat": SCORECARD_CAVEAT,
        "instrument": {
            "id": "collective_influence",
            "kind": "causal",
            "delta": SCORECARD_DELTA,
            "description": ("responsiveness of per-capita harvest to a "
                            "collective downward ask-shift at t=0; paired "
                            "same-key rollouts"),
        },
        "metrics": [
            {"id": "influence_preserved", "label": "influence preserved",
             "kind": "causal"},
            {"id": "stock_pct", "label": "final stock (fraction of capacity)",
             "kind": "descriptive"},
            {"id": "compliance_rate", "label": "compliance rate",
             "kind": "descriptive"},
            {"id": "influence_fidelity", "label": "correlational fidelity",
             "kind": "descriptive"},
        ],
        "rows": rows,
    }


# --- influence-curve.json -------------------------------------------------------

def build_influence_curve(n_steps: int, n_seeds: int, t0s, window: int) -> dict:
    """Influence at increasing intervention times on the page's one substrate:
    responsiveness of the post-shift harvest window to a one-shot collective
    ask-shift scheduled at t0 (wp3's responsiveness pattern).

    What this substrate actually shows (measured 2026-08-07, 16 seeds): the
    defended commons responds at EVERY t0 (~+0.49 throughout); the undefended
    one never meaningfully responds — zero once the stock has collapsed
    (t0>=30) and slightly sign-perverse at t0=0 (~-0.06: at the knife-edge a
    lower ask makes the stock last longer, so the ecology governs outcomes,
    not the asks — the whitepaper's scenario-1 lesson). There is no gradual
    decay window here; the from-birth-vs-now GAP story belongs to substrates
    with slow capture (compute_economy), not to a fast-collapsing commons."""
    never = n_steps + 1

    @transform(reads=["principal_pref", "vote"],
               writes=["principal_pref", "vote"])
    def shift_now(state):
        return shift_preferences(state, CURVE_DELTA)

    conditions = {}
    for name in CURVE_CONDITIONS:
        mechs = _mechanisms_for(CONDITIONS[name])
        base = make_env(ENV, mechanisms=mechs)
        means, ses = [], []
        for t0 in t0s:
            intervened = make_env(ENV, mechanisms=(
                *mechs, scheduled(shift_now, cadence=never,
                                  phase_offset=t0 % never, onset=t0)))
            lo, hi = t0 + 1, min(t0 + 1 + window, n_steps)

            def outcome(tr, lo=lo, hi=hi):
                return jnp.mean(tr["harvest"][lo:hi])

            response = intervention_response(
                base, intervened, jr.PRNGKey(SEED0), n_seeds, n_steps,
                outcome_fn=outcome, scale=CURVE_DELTA)
            stats = _stats(response)
            means.append(stats["mean"])
            ses.append(stats["se"])
        conditions[name] = {"mean": means, "se": ses}

    return {
        "env": ENV, "T": n_steps, "n_seeds": n_seeds, "seed0": SEED0,
        "delta": CURVE_DELTA, "window": window, "t0": list(t0s),
        "caveat": SCORECARD_CAVEAT,
        "instrument": {
            "id": "intervention_response",
            "kind": "causal",
            "description": ("responsiveness of the mean harvest over the "
                            f"{window} ticks after t0 to a one-shot collective "
                            "ask-shift scheduled at t0; paired same-key "
                            "batches per condition"),
        },
        "conditions": conditions,
    }


# --- graph-matrix.json ----------------------------------------------------------

def build_graph_matrix() -> dict:
    """A real agent-agent adjacency and its spectrum, for the page's closing
    arc: any graph is a matrix, one step is one matrix multiply, and the
    Laplacian spectrum reads the structure. Source: value_contagion's
    friendship graph (the same graph fiedler_partition_alignment ships
    against). Spectral conventions mirror the engine's spectral family
    exactly: L = D - W, eigh ascending, Fiedler = column 1; the alignment
    score is computed BY the engine function, not re-derived here."""
    env = make_env("value_contagion")
    state = value_contagion.make_state(env.config, jr.PRNGKey(SEED0))
    W = np.asarray(state.adj_matrices["friendship"], dtype=np.float64)
    types = np.asarray(state.node_types)
    n = W.shape[0]

    L = np.diag(W.sum(axis=-1)) - W
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    alignment = float(fiedler_partition_alignment_of(
        state.adj_matrices["friendship"], state.node_types))

    return {
        "env": "value_contagion",
        "layer": "friendship",
        "N": int(n),
        "seed0": SEED0,
        "params": {k: v for k, v in dataclasses.asdict(env.config).items()
                   if isinstance(v, (int, float, bool, str))},
        "node_types": [int(t) for t in types],
        "adj": [round(float(v), ROUND_DECIMALS) for v in W.ravel()],
        "spectral": {
            "eigenvalues": [round(float(v), ROUND_DECIMALS)
                            for v in eigenvalues],
            "fiedler": [round(float(v), ROUND_DECIMALS)
                        for v in eigenvectors[:, 1]],
            "spectral_gap": round(float(eigenvalues[1]), ROUND_DECIMALS),
            "fiedler_alignment": round(alignment, ROUND_DECIMALS),
        },
    }


# --- the export -----------------------------------------------------------------

def export_fixtures(out: str, n_steps: int, validators, smoke: bool = False) -> dict:
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    sizes = {}

    def write(rel, obj, kind):
        validators[kind].validate(obj)
        path = os.path.join(out, rel)
        with open(path, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        sizes[rel.replace(os.sep, "/")] = os.path.getsize(path)
        assert sizes[rel.replace(os.sep, "/")] <= MAX_FILE_BYTES, \
            f"{rel} is {sizes[rel.replace(os.sep, '/')]} B > {MAX_FILE_BYTES} B gate"

    write("pipeline-subsets.json", build_subsets_fixture(), "subsets")
    write("system-graphs.json", build_systems_fixture(), "systems")
    write("snippets.json", extract_snippets(), "snippets")
    write("schedule-golden.json", build_schedule_golden(), "scheduleGolden")
    write("state-shapes.json", build_state_shapes(), "stateShapes")
    write("graph-matrix.json", build_graph_matrix(), "graphMatrix")
    for condition in CONDITIONS:
        payload = build_run_payload(condition, n_steps)
        for name, arr in payload["global"].items():
            assert len(arr) == n_steps, f"{condition}: {name} global length"
        assert not payload["node"], f"{condition}: node section must stay empty"
        write(os.path.join("runs", f"{ENV}.{condition}.json"), payload, "run")

    scorecard = build_scorecard(n_steps, 2 if smoke else SCORECARD_SEEDS)
    write("scorecard.json", scorecard, "scorecard")
    curve = build_influence_curve(
        n_steps, 2 if smoke else CURVE_SEEDS,
        (0, 1) if smoke else CURVE_T0S, 2 if smoke else CURVE_WINDOW)
    write("influence-curve.json", curve, "influenceCurve")

    if not smoke:
        # Committed expectations, gd_bundles-style: the ordering claims the
        # page narrates must hold in the shipped fixtures, or the export
        # fails — a miss is reported and investigated, never retuned away.
        by_cond = {r["condition"]: r["values"] for r in scorecard["rows"]}
        assert by_cond["graduated_sanctions"]["stock_pct"]["mean"] \
            > by_cond["baseline"]["stock_pct"]["mean"], \
            "expectation: sanctions sustain the stock over baseline"
        assert by_cond["graduated_sanctions"]["influence_preserved"]["mean"] \
            > by_cond["baseline"]["influence_preserved"]["mean"], \
            "expectation: sanctions preserve causal influence over baseline"
        # Expectation corrected 2026-08-07 after first full run: the original
        # guess ("undefended influence decays with t0") assumed a gradual
        # decay this substrate does not produce — undefended influence is
        # ~zero at every t0 (collapse is fast; at t0=0 the ecology binds and
        # the reading is slightly sign-perverse). Numbers were not retuned;
        # the expectations now state the claims the page narrates.
        base_curve = curve["conditions"]["baseline"]["mean"]
        defended_curve = curve["conditions"]["graduated_sanctions"]["mean"]
        assert min(defended_curve) > 0, \
            "expectation: the defended commons responds at every t0"
        assert max(abs(b) for b in base_curve) < 0.5 * min(defended_curve), \
            "expectation: the undefended commons is never meaningfully responsive"
        assert defended_curve[-1] > base_curve[-1], \
            "expectation: late influence survives only defended"

    # the shared record core hosts the explainer schema's fields; the scorecard
    # and influence-curve designs are part of the address
    core = runs.manifest(
        env=ENV, config=make_env(ENV).config, T=n_steps, seed0=SEED0,
        n_seeds=2 if smoke else SCORECARD_SEEDS, kind="library_explainer",
        conditions=CONDITIONS, metrics=make_env(ENV).metrics,
        extra={"scorecard": {"delta": SCORECARD_DELTA},
               "influence_curve": {"n_seeds": 2 if smoke else CURVE_SEEDS,
                                   "delta": CURVE_DELTA,
                                   "t0s": list((0, 1) if smoke else CURVE_T0S),
                                   "window": 2 if smoke else CURVE_WINDOW,
                                   "conditions": list(CURVE_CONDITIONS)}})
    manifest = {
        **core,
        "schema_version": "1",
        "kind": "library_explainer",
        "contract_version": "1.1",
        "env": ENV,
        "T": n_steps,
        "seed0": SEED0,
        "conditions": sorted(CONDITIONS),
        "run_whitelist": list(RUN_WHITELIST),
    }
    runs.write(out, manifest, include=sorted(sizes),
               validate=validators["manifest"].validate, overwrite=True)

    total = sum(os.path.getsize(os.path.join(root, name))
                for root, _, names in os.walk(out) for name in names)
    assert total <= MAX_TOTAL_BYTES, \
        f"fixture set totals {total} B > {MAX_TOTAL_BYTES} B gate"
    print(f"library_explainer: {len(sizes) + 1} files, total {total / 1e3:.0f} KB")
    return {"total": total, "files": len(sizes) + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    parser.add_argument("--out", default=None,
                        help=f"output dir (default {DEFAULT_OUT})")
    args = parser.parse_args()

    n_steps = 5 if args.smoke else T
    out = args.out or (os.path.join(tempfile.gettempdir(),
                                    "library_explainer_smoke")
                       if args.smoke else DEFAULT_OUT)
    export_fixtures(out, n_steps, _validators(), smoke=args.smoke)

    # the schema ships with the fixtures so the site validates the same document
    with open(SCHEMA_PATH) as f:
        schema_text = f.read()
    with open(os.path.join(out, "explainer.schema.json"), "w") as f:
        f.write(schema_text)
    print(f"fixtures written to {out}")


if __name__ == "__main__":
    main()
