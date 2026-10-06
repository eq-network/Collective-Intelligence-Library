"""The run store: a record round-trips, names its files, and its hash is an address."""
import json

import pytest

from cilib.core.schedule import ScheduleSpec
from cilib.environments import make_env
from cilib.mechanisms import QuotaVoteConfig, SanctionConfig
from cilib import runs

TRIPLES = [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5)),
           ("graduated_sanction", SanctionConfig(), None)]


def _manifest(**over):
    env = make_env("governed_commons")
    base = dict(env="governed_commons", config=env.config, T=12, seed0=0, n_seeds=2,
                kind="scorecard", conditions={"baseline": [], "defended": TRIPLES},
                metrics=env.metrics)
    return runs.manifest(**{**base, **over})


def test_manifest_carries_address_metrics_and_engine():
    m = _manifest()
    assert m["schema_version"] == "1"
    first = m["address"]["conditions"]["defended"][0]
    assert first["name"] == "quota_vote"
    assert first["config"] == {"quantile": QuotaVoteConfig().quantile}
    assert first["schedule"] == {"cadence": 5, "phase_offset": 0, "onset": 0}
    assert m["address"]["conditions"]["defended"][1]["schedule"] is None
    assert {e["id"] for e in m["metrics"]} == set(make_env("governed_commons").metrics)
    assert m["engine"]["version"] and m["engine"]["git_rev"] and m["engine"]["source"]
    assert m["created"].endswith("+00:00")
    json.dumps(m)  # nothing non-serializable slipped in


def test_config_hash_is_an_address():
    assert _manifest()["config_hash"] == _manifest()["config_hash"]
    assert _manifest()["config_hash"] != _manifest(T=13)["config_hash"]
    assert _manifest()["config_hash"] != _manifest(seed0=1)["config_hash"]
    assert _manifest()["config_hash"] != _manifest(conditions={"baseline": []})["config_hash"]
    assert _manifest()["config_hash"] != _manifest(overrides={"K_cap": 250.0})["config_hash"]
    assert _manifest()["config_hash"] != _manifest(extra={"axes": [["K_cap", [1, 2]]]})["config_hash"]
    # what did not change the design does not change the hash
    assert _manifest()["config_hash"] == _manifest(notes="x", kind="other")["config_hash"]


def test_write_read_roundtrip_and_checksums(tmp_path):
    m = _manifest()
    out = runs.write(runs.run_dir(tmp_path, "commons", m["created"]), m,
                     files={"scalars.json": {"stock_pct": [0.5, 0.4]},
                            "runs/c-0.json": {"x": 1}})
    assert out.parent == tmp_path and out.name.endswith("_commons")
    back = runs.read(out)
    assert set(back["checksums"]) == {"scalars.json", "runs/c-0.json"}
    assert back["config_hash"] == m["config_hash"]
    assert json.loads((out / "scalars.json").read_text()) == {"stock_pct": [0.5, 0.4]}


def test_only_named_files_join_the_record(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "leftover.json").write_text("{}")
    (tmp_path / "r" / "mine.json").write_text("{}")
    out = runs.write(tmp_path / "r", _manifest(), files={"s.json": {}}, include=["mine.json"])
    assert set(runs.read(out)["checksums"]) == {"s.json", "mine.json"}


def test_a_record_is_written_once(tmp_path):
    runs.write(tmp_path / "r", _manifest())
    with pytest.raises(FileExistsError):
        runs.write(tmp_path / "r", _manifest())
    runs.write(tmp_path / "r", _manifest(), overwrite=True)


def test_read_detects_a_tampered_or_missing_file(tmp_path):
    out = runs.write(tmp_path / "r", _manifest(), files={"scalars.json": {"a": 1}})
    (out / "scalars.json").write_text('{"a": 2}')
    with pytest.raises(ValueError, match="checksum"):
        runs.read(out)
    (out / "scalars.json").unlink()
    with pytest.raises(FileNotFoundError):
        runs.read(out)


def test_read_detects_a_tampered_address(tmp_path):
    out = runs.write(tmp_path / "r", _manifest())
    m = json.loads((out / "manifest.json").read_text())
    m["address"]["T"] = 999
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="address"):
        runs.read(out)


def test_missing_cache_arrays_are_allowed(tmp_path):
    out = runs.write(tmp_path / "r", _manifest(), files={"snapshot.npz": b"\x00\x01"})
    (out / "snapshot.npz").unlink()
    assert "snapshot.npz" in runs.read(out)["checksums"]


def test_index_lists_records_newest_first(tmp_path):
    a = _manifest(); a["created"] = "2026-01-01T00:00:00+00:00"
    b = _manifest(); b["created"] = "2026-02-01T00:00:00+00:00"
    runs.write(tmp_path / "a", a)
    runs.write(tmp_path / "nested" / "b", b)
    idx = runs.index(tmp_path)
    assert [m["path"] for m in idx] == ["nested/b", "a"]


def test_validate_hook_runs_on_the_final_manifest(tmp_path):
    seen = {}
    runs.write(tmp_path / "r", _manifest(), files={"s.json": {}},
               validate=lambda m: seen.update(m))
    assert "s.json" in seen["checksums"]


def test_a_traced_onset_round_trips(tmp_path):
    """``ScheduleSpec(onset="t_on_now")`` names a global_attrs dial; the record keeps
    the name, and the name enters the address like an int onset does."""
    sched = [("quota_vote", QuotaVoteConfig(), ScheduleSpec(onset="t_on_now"))]
    m = _manifest(conditions={"defended": sched})
    out = runs.write(tmp_path / "r", m)
    back = runs.read(out)["address"]["conditions"]["defended"][0]["schedule"]
    assert back == {"cadence": 1, "phase_offset": 0, "onset": "t_on_now"}
    other = [("quota_vote", QuotaVoteConfig(), ScheduleSpec(onset=0))]
    assert m["config_hash"] != _manifest(conditions={"defended": other})["config_hash"]
