"""Behavioral tests for the /library/prototype fixture exporter.

Direction and structure, not bit-exact numbers: the fixtures are a boundary
crossing, so what must hold is that they faithfully carry what the engine
derives — batches from compile_pipeline, hazard edges from declared effects,
payloads within the contract and the size gates.
"""
import json
import os

import pytest

from cilib import runs
from cilib.core.pipeline import get_execution_order
from experiments.library_explainer import export as ex
from experiments.library_explainer.config import CONDITIONS, ENV


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    """One smoke export per module (T=5); returns (out_dir, loaded JSONs)."""
    out = str(tmp_path_factory.mktemp("library_explainer"))
    ex.export_fixtures(out, n_steps=5, validators=ex._validators(), smoke=True)

    def load(rel):
        with open(os.path.join(out, rel)) as f:
            return json.load(f)
    loaded = {"manifest": load("manifest.json"),
              "subsets": load("pipeline-subsets.json"),
              "systems": load("system-graphs.json"),
              "snippets": load("snippets.json"),
              "golden": load("schedule-golden.json"),
              "shapes": load("state-shapes.json"),
              "scorecard": load("scorecard.json"),
              "curve": load("influence-curve.json"),
              "matrix": load("graph-matrix.json"),
              "runs": {c: load(os.path.join("runs", f"{ENV}.{c}.json"))
                       for c in CONDITIONS}}
    return out, loaded


def test_manifest_checksums_match_files(fixtures):
    out, loaded = fixtures
    checksums = loaded["manifest"]["checksums"]
    assert set(checksums) == {
        "pipeline-subsets.json", "system-graphs.json", "snippets.json",
        "schedule-golden.json", "state-shapes.json", "scorecard.json",
        "influence-curve.json", "graph-matrix.json",
        *(f"runs/{ENV}.{c}.json" for c in CONDITIONS)}
    for rel, digest in checksums.items():
        assert runs.sha256_file(os.path.join(out, rel)) == digest, rel


def test_subsets_rows_complete_and_bitmask_indexed(fixtures):
    _, loaded = fixtures
    subsets = loaded["subsets"]
    n = len(subsets["transforms"])
    assert len(subsets["rows"]) == 2 ** n
    for mask, row in enumerate(subsets["rows"]):
        assert row["enabled"] == [i for i in range(n) if mask >> i & 1]
        assert sorted(i for batch in row["batches"] for i in batch) \
            == row["enabled"]


def test_full_row_batches_match_engine(fixtures):
    """The fixture's full-set row must equal a fresh get_execution_order over
    the really-built steps — the page's BatchBoard is exactly this lookup."""
    _, loaded = fixtures
    full = loaded["subsets"]["rows"][-1]
    steps, _entries = ex.subset_transforms()
    pos = {id(t): i for i, t in enumerate(steps)}
    engine_batches = [[pos[id(t)] for t in batch]
                      for batch in get_execution_order(list(steps))]
    assert engine_batches == full["batches"]


def test_scene_claims_hold(fixtures):
    """The didactic facts the BatchBoard scene points at, asserted as
    direction/ordering (a change in declared effects should fail here)."""
    _, loaded = fixtures
    subsets = loaded["subsets"]
    names = [t["name"] for t in subsets["transforms"]]
    assert names == ["harvest", "regrow", "quota_vote", "graduated_sanction"]
    full = subsets["rows"][-1]
    depth = {i: b for b, batch in enumerate(full["batches"]) for i in batch}

    # quota_vote's write (policy_target) is what graduated_sanction reads:
    # the vote must run in an earlier batch than the sanction.
    assert depth[names.index("quota_vote")] < depth[names.index("graduated_sanction")]
    # regrow and quota_vote share no field — they run in the same batch, in
    # parallel. This is the scene's claim.
    assert depth[names.index("regrow")] == depth[names.index("quota_vote")]
    # harvest opens the round.
    assert depth[names.index("harvest")] == 0

    edge_kinds = {(i, j): kinds for i, j, kinds in full["edges"]}
    hr = (names.index("harvest"), names.index("regrow"))
    assert "RAW" in edge_kinds[hr] and "WAW" in edge_kinds[hr]
    assert (names.index("regrow"), names.index("quota_vote")) not in edge_kinds


def test_singletons_and_empty_row(fixtures):
    _, loaded = fixtures
    rows = loaded["subsets"]["rows"]
    n = len(loaded["subsets"]["transforms"])
    assert rows[0] == {"enabled": [], "edges": [], "batches": []}
    for i in range(n):
        row = rows[1 << i]
        assert row["enabled"] == [i]
        assert row["batches"] == [[i]] and row["edges"] == []


def test_systems_track_mechanism_toggles(fixtures):
    """Toggling a mechanism adds/removes its transform node — the System
    view's claim, checked at the fixture level."""
    _, loaded = fixtures
    conds = loaded["systems"]["conditions"]
    assert set(conds) == set(CONDITIONS)

    def transform_ids(cond):
        return {n["id"] for n in conds[cond]["nodes"] if n["kind"] == "transform"}

    base, quota, sanct = (transform_ids(c) for c in
                          ("baseline", "quota_voting", "graduated_sanctions"))
    assert base < quota < sanct
    assert any("quota_vote" in t for t in quota - base)
    assert any("graduated_sanction" in t for t in sanct - quota)


def test_runs_are_lean_and_complete(fixtures):
    _, loaded = fixtures
    whitelist = set(loaded["manifest"]["run_whitelist"])
    for cond, payload in loaded["runs"].items():
        assert payload["node"] == {}, cond
        assert set(payload["global"]) == whitelist, cond
        T = payload["meta"]["T"]
        for name, arr in payload["global"].items():
            assert len(arr) == T, (cond, name)
        assert payload["meta"]["params"]["condition"] == cond
        assert "adj" not in payload and "system" not in payload


def test_schedule_golden_matches_closed_form(fixtures):
    """The golden came from the engine's scheduled() wrapper; the closed form
    is the paper's definition. They must agree tick for tick."""
    _, loaded = fixtures
    golden = loaded["golden"]
    for combo in golden["combos"]:
        c, p, o = combo["cadence"], combo["phase_offset"], combo["onset"]
        for t, fired in enumerate(combo["fires"]):
            expected = t >= o and (t - p) % c == 0
            assert fired == expected, f"(c={c}, p={p}, o={o}) at t={t}"


def test_snippets_resolve_and_hash(fixtures):
    import hashlib
    from experiments.library_explainer.snippets import SNIPPETS
    _, loaded = fixtures
    snippets = loaded["snippets"]
    assert set(snippets) == set(SNIPPETS)
    for sid, snippet in snippets.items():
        token = SNIPPETS[sid][3]
        assert token in snippet["text"], (sid, token)
        assert snippet["sha256"] == hashlib.sha256(
            snippet["text"].encode("utf-8")).hexdigest(), sid
        assert snippet["end_line"] >= snippet["start_line"], sid


def test_scorecard_and_curve_structure(fixtures):
    """Structure only at smoke scale — the ordering claims are committed
    expectations enforced by the full export, gd_bundles-style."""
    _, loaded = fixtures
    scorecard = loaded["scorecard"]
    assert [r["condition"] for r in scorecard["rows"]] == list(CONDITIONS)
    metric_ids = {m["id"] for m in scorecard["metrics"]}
    for row in scorecard["rows"]:
        assert set(row["values"]) == metric_ids
    assert scorecard["caveat"]
    assert scorecard["instrument"]["kind"] == "causal"

    curve = loaded["curve"]
    for series in curve["conditions"].values():
        assert len(series["mean"]) == len(curve["t0"])
        assert len(series["se"]) == len(curve["t0"])


def test_state_shapes_describe_the_population(fixtures):
    _, loaded = fixtures
    shapes = loaded["shapes"]
    N = shapes["N"]
    assert N >= 1
    for name, desc in shapes["fields"]["node_attrs"].items():
        assert desc["shape"][0] == N, name


def test_graph_matrix_spectrum(fixtures):
    """The fixture's spectrum must be the engine's reading of the engine's
    graph: L = D - W conventions, connected graph, alignment recomputed by
    the shipped metric function."""
    import numpy as np
    import jax.numpy as jnp
    from cilib.metrics.families.spectral import fiedler_partition_alignment_of

    _, loaded = fixtures
    gm = loaded["matrix"]
    n = gm["N"]
    assert len(gm["adj"]) == n * n
    assert len(gm["node_types"]) == n
    eig = gm["spectral"]["eigenvalues"]
    assert len(eig) == n and len(gm["spectral"]["fiedler"]) == n
    assert abs(eig[0]) < 1e-3                     # lambda_1 ~ 0 for L = D - W
    assert gm["spectral"]["spectral_gap"] > 0     # connected friendship graph
    assert eig == sorted(eig)

    W = jnp.asarray(np.asarray(gm["adj"], dtype=np.float64).reshape(n, n))
    recomputed = float(fiedler_partition_alignment_of(
        W, jnp.asarray(gm["node_types"])))
    assert abs(recomputed - gm["spectral"]["fiedler_alignment"]) < 5e-3


def test_size_gates(fixtures):
    out, _ = fixtures
    from experiments.library_explainer.config import (MAX_FILE_BYTES,
                                                      MAX_TOTAL_BYTES)
    total = 0
    for root, _dirs, files in os.walk(out):
        for name in files:
            size = os.path.getsize(os.path.join(root, name))
            assert size <= MAX_FILE_BYTES, name
            total += size
    assert total <= MAX_TOTAL_BYTES
