"""Write the sweep bundles: manifest.json + scalars.json + runs/c-*.json.

    python -m experiments.gd_bundles.export [--smoke] [--bundle ID] [--out DIR]

Consumes run.py's results.json (scalar lattice with CIs) and re-runs the
playback cells single-seed, deterministically (representative run =
``env.run(PRNGKey(seed0))``), emitting contract-v1.1 payloads through
``cilib.environments.webexport.trajectory_payload`` with the bundle's
whitelist, engine-side derived series, mandatory rounding, and no ``adj``.

Every bundle is validated on write against schema/bundle.schema.json (Python
``jsonschema`` — install via ``pip install -e .[export]``) plus the code-level
checks the schema documents (array lengths, checksums, size gates). eq-network
CI re-validates the same artifacts with ajv — the two independent validators
of one shared schema are the drift guard.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import tempfile

import jax
import jax.random as jr
import numpy as np

from cilib.environments import (
    capital_economy, delegative_polity, influence_exchange, ledger_society,
    make_env,
)
from cilib.environments.system_graph import system_graph
from cilib.environments.webexport import trajectory_payload
from cilib import runs
from cilib.metrics.identity import metric_id

from .config import (
    BUNDLES, MAX_BUNDLE_BYTES, MAX_RUN_BYTES, ROUND_DECIMALS, SEED0,
    BundleSpec,
)
from .derived import DERIVED
from .overlays import overlays_for
from .run import RESULTS, SMOKE_RESULTS, cell_indices, cell_overrides, smoke_spec

ENV_MODULES = {
    "capital_economy": capital_economy,
    "influence_exchange": influence_exchange,
    "delegative_polity": delegative_polity,
    "ledger_society": ledger_society,
}
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema", "bundle.schema.json")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "dist-bundles")


def _validators():
    try:
        import jsonschema
    except ImportError as e:                       # pragma: no cover
        raise SystemExit("bundle validation needs jsonschema: "
                         "pip install -e .[export]") from e
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    def against(def_name):
        return jsonschema.Draft202012Validator(
            {"$ref": f"#/$defs/{def_name}", "$defs": schema["$defs"]})
    return {k: against(k) for k in ("manifest", "scalars", "run")}


def _playback_cells(spec: BundleSpec):
    if spec.playback == "all":
        return list(cell_indices(spec))
    return [tuple(c) for c in spec.playback]


def default_cell_of(spec: BundleSpec, base_config: dict, strict: bool):
    """The base config's cell on the lattice. The house rule 'the default is a
    nothing-special point on the swept curve' becomes a structural export
    failure here (it would have caught regime_rate=0.10 sitting off-lattice).
    Smoke twins truncate axes, so strict=False falls back to cell 0."""
    cell = []
    for a in spec.axes:
        match = next((i for i, x in enumerate(a.values)
                      if abs(x - base_config[a.name]) < 1e-9), None)
        if match is None:
            assert not strict, (
                f"{spec.bundle_id}: default {a.name}={base_config[a.name]} "
                f"is not on the swept axis {list(a.values)}")
            match = 0
        cell.append(match)
    if strict and spec.playback != "all":
        assert tuple(cell) in {tuple(c) for c in spec.playback}, \
            f"{spec.bundle_id}: default cell {cell} has no playback run"
    return cell


def export_bundle(spec: BundleSpec, lattice: dict, out_root: str,
                  n_steps: int, validators, strict: bool = True) -> dict:
    out = os.path.join(out_root, spec.bundle_id)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    shape = lattice["shape"]
    n_cells = int(np.prod(shape))
    assert len(lattice["cells"]) == n_cells, "lattice incomplete"

    # --- scalars.json: flatten rows into row-major arrays ------------------------
    order = {tuple(r["cell"]): r["values"] for r in lattice["cells"]}
    flat_cells = list(itertools.product(*(range(n) for n in shape)))
    scalars = {
        "shape": shape, "order": "row-major",
        "metrics": {m.id: {k: [order[c][m.id][k] for c in flat_cells]
                           for k in ("point", "lo", "hi")}
                    for m in spec.metrics},
        "has_run": [1] * n_cells,
    }
    validators["scalars"].validate(scalars)
    for m in spec.metrics:
        assert all(len(scalars["metrics"][m.id][k]) == n_cells for k in ("point", "lo", "hi"))
    with open(os.path.join(out, "scalars.json"), "w") as f:
        json.dump(scalars, f, separators=(",", ":"))

    # --- runs/: playback cells, representative single run ------------------------
    mod = ENV_MODULES[spec.env]
    sizes, trace_field_kinds = {}, {"global": set(), "node": set(), "static": set()}
    for cell in _playback_cells(spec):
        overrides = {**spec.overrides, **cell_overrides(spec, cell)}
        env = make_env(spec.env, **overrides)
        finals, trace = env.run(jr.PRNGKey(SEED0), n_steps)
        derived_all = DERIVED[spec.env](trace, env.config)
        missing = set(spec.derived) - set(derived_all)
        assert not missing, f"derived fns missing {missing} for {spec.env}"
        payload = trajectory_payload(
            trace, finals, game_id=spec.env, n_steps=n_steps, seed=SEED0,
            params={**dataclasses.asdict(env.config)},
            scalars=env.evaluate(trace),
            system=system_graph(mod.build_steps(env.config, ()),
                                mod.make_state(env.config, jr.PRNGKey(SEED0))),
            whitelist=spec.whitelist,
            derived_global={k: derived_all[k] for k in spec.derived},
            round_decimals=ROUND_DECIMALS, include_adj=False)
        validators["run"].validate(payload)
        T_, N_ = payload["meta"]["T"], payload["meta"]["N"]
        for name, arr in payload["global"].items():
            assert len(arr) == T_, f"{name} global length"
        for name, arr in payload["node"].items():
            assert len(arr) == T_ * N_, f"{name} node length"
        for kind in trace_field_kinds:
            trace_field_kinds[kind] |= set(payload[kind])

        rel = os.path.join("runs", "c-" + "-".join(map(str, cell)) + ".json")
        path = os.path.join(out, rel)
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        sizes[rel] = os.path.getsize(path)
        assert sizes[rel] <= MAX_RUN_BYTES, \
            f"{spec.bundle_id}/{rel} is {sizes[rel]} B > {MAX_RUN_BYTES} B gate"

    # --- manifest.json (written last: carries the checksums) ---------------------
    base_env = make_env(spec.env, **spec.overrides)
    axes = [dataclasses.asdict(a) for a in spec.axes]
    for a in axes:
        a["values"] = list(a["values"])
        for opt in ("unit", "anchor", "paper_ref"):
            if a.get(opt) is None:
                a.pop(opt, None)
    # the shared record core (address block, config_hash, engine, created) hosts
    # the bundle schema's own fields; the sweep axes are part of the address
    core = runs.manifest(
        env=spec.env, config=base_env.config, overrides=spec.overrides,
        T=n_steps, seed0=SEED0, n_seeds=lattice.get("n_seeds", 0), kind="bundle",
        extra={"axes": [[a["name"], a["values"]] for a in axes]},
        metrics={m.id: base_env.metrics[m.id] for m in spec.metrics}, notes=spec.notes)
    manifest = {
        **core,
        "schema_version": "2",
        "contract_version": "1.1",
        "bundle_id": spec.bundle_id,
        "env": spec.env,
        "suite": "gradual_disempowerment",
        "base_config": dataclasses.asdict(base_env.config),
        "overrides": spec.overrides,
        "mechanisms": [],
        "T": n_steps,
        "n_seeds": lattice.get("n_seeds", 0),
        "seed0": SEED0,
        "representative_seed_index": 0,
        "axes": axes,
        "default_cell": default_cell_of(
            spec, dataclasses.asdict(base_env.config), strict),
        "overlays": overlays_for(spec),
        "metrics": [{"label": m.label, "direction": m.direction, "late_window": 0.25,
                     **metric_id(m.id, base_env.metrics[m.id]).as_dict()}
                    for m in spec.metrics],
        "trace_fields": {
            "global": sorted(trace_field_kinds["global"] - set(spec.derived)),
            "node": sorted(trace_field_kinds["node"]),
            "static": sorted(trace_field_kinds["static"]),
            "derived": list(spec.derived),
        },
        "playback": {"cells": ("all" if spec.playback == "all"
                               else [list(c) for c in spec.playback]),
                     "tick_stride": 1},
        "checksums": {},
    }
    runs.write(out, manifest, include=["scalars.json", *sizes],
               validate=validators["manifest"].validate, overwrite=True)

    total = sum(os.path.getsize(os.path.join(root, name))
                for root, _, names in os.walk(out) for name in names)
    assert total <= MAX_BUNDLE_BYTES, \
        f"{spec.bundle_id} totals {total} B > {MAX_BUNDLE_BYTES} B gate"
    biggest = max(sizes.values()) if sizes else 0
    print(f"{spec.bundle_id}: {len(sizes)} playback cells, "
          f"biggest run {biggest / 1e3:.0f} KB, bundle total {total / 1e6:.2f} MB")
    return {"total": total, "runs": len(sizes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--out", default=None,
                        help=f"output root (default {DEFAULT_OUT})")
    args = parser.parse_args()

    results_path = SMOKE_RESULTS if args.smoke else RESULTS
    if not os.path.exists(results_path):
        raise SystemExit(f"no {results_path} — run experiments.gd_bundles.run first"
                         + (" --smoke" if args.smoke else ""))
    with open(results_path) as f:
        results = json.load(f)

    out_root = args.out or (os.path.join(tempfile.gettempdir(), "gd_bundles_smoke")
                            if args.smoke else DEFAULT_OUT)
    validators = _validators()
    specs = [s for s in BUNDLES if args.bundle in (None, s.bundle_id)]
    for spec in specs:
        spec_x = smoke_spec(spec) if args.smoke else spec
        lattice = dict(results["bundles"][spec.bundle_id],
                       n_seeds=results["n_seeds"])
        export_bundle(spec_x, lattice, out_root, lattice["T"], validators,
                      strict=not args.smoke)

    # the schema ships with the bundles so the site validates the same document
    with open(SCHEMA_PATH) as f:
        schema_text = f.read()
    with open(os.path.join(out_root, "bundle.schema.json"), "w") as f:
        f.write(schema_text)
    print(f"bundles written to {out_root}")


if __name__ == "__main__":
    main()
