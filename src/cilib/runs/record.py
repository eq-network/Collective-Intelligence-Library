"""The run store (observability O1): a directory per record, a manifest written last.

A record is a directory holding ``manifest.json`` plus whatever files the caller
hands over (scalar series, a scorecard, a trajectory payload). The manifest names
every other file with its sha256, so a record is self-describing and a stale
copy is detectable. Manifests and scalar series are small and belong in git;
raw arrays (``*.npz``) inside a record stay ignored and must be re-derivable.

Not a database: directories, JSON, npz. The index is derived from the manifests
on read, never stored, so it cannot drift.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from cilib.core.schedule import ScheduleSpec
from cilib.metrics.identity import metric_ids
from .provenance import engine_info

SCHEMA_VERSION = "1"
MANIFEST = "manifest.json"
PathLike = Union[str, "os.PathLike[str]"]


# --- hashing -----------------------------------------------------------------

def canonical(obj: Any) -> str:
    """Key-sorted, whitespace-free JSON: the same object always hashes the same."""
    return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(obj: Any) -> Any:
    """Dataclasses, tuples and array scalars into plain JSON values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):          # numpy / jax scalars and arrays
        return obj.tolist()
    return obj


# --- manifest ------------------------------------------------------------------

MechanismTriple = Tuple[str, Any, Optional[ScheduleSpec]]


def mechanism_records(triples: Iterable[MechanismTriple]) -> List[dict]:
    """Serialize ``(catalog key, config, ScheduleSpec | None)`` triples, the one
    convention every condition table in this repo already uses."""
    return [{"name": name,
             "config": jsonable(cfg),
             "schedule": None if sched is None else dataclasses.asdict(sched)}
            for name, cfg, sched in triples]


def address_hash(m: dict) -> str:
    """sha256 of the manifest's ``address`` block, canonically serialized."""
    return sha256_text(canonical(m["address"]))


def manifest(*, env: str, config: Any, T: int, seed0: int, n_seeds: int,
             kind: str = "run",
             mechanisms: Optional[Iterable[MechanismTriple]] = None,
             conditions: Optional[Dict[str, Iterable[MechanismTriple]]] = None,
             overrides: Optional[dict] = None,
             extra: Optional[dict] = None,
             metrics: Optional[Dict[str, Callable]] = None,
             notes: str = "") -> dict:
    """Build a manifest. The ``address`` block (everything a replay needs) is hashed
    into ``config_hash``; engine and timestamp are recorded beside it, not in it,
    so the hash names an experimental design, not a payload or a software version.
    A schema-bound writer spreads its own fields over this dict; the block survives.

    ``mechanisms`` describes one condition, ``conditions`` a comparison of several;
    ``extra`` adds design fields the fixed set lacks (a sweep's axes); a record is
    one experimental design over ``n_seeds`` seeds from ``seed0``.
    """
    address = {
        "env": env,
        "config": jsonable(config),
        "overrides": jsonable(dict(overrides or {})),
        "mechanisms": mechanism_records(mechanisms or ()),
        "conditions": {name: mechanism_records(tr)
                       for name, tr in (conditions or {}).items()},
        "T": int(T), "seed0": int(seed0), "n_seeds": int(n_seeds),
        **jsonable(dict(extra or {})),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "address": address,
        "config_hash": sha256_text(canonical(address)),
        "metrics": metric_ids(metrics or {}),
        "engine": engine_info(),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "checksums": {},
    }


# --- store ---------------------------------------------------------------------

def run_dir(root: PathLike, name: str, created: Optional[str] = None) -> Path:
    """``<root>/<UTC stamp>_<name>``; pass the manifest's ``created`` to keep them equal."""
    when = (_dt.datetime.fromisoformat(created) if created
            else _dt.datetime.now(_dt.timezone.utc))
    return Path(root) / f"{when.strftime('%Y%m%dT%H%M%SZ')}_{name}"


def write(out: PathLike, manifest: dict,
          files: Optional[Dict[str, Any]] = None,
          include: Iterable[str] = (),
          validate: Optional[Callable[[dict], None]] = None,
          overwrite: bool = False) -> Path:
    """Write ``files`` (relative path -> JSON-able object, or bytes), then the manifest
    last with a checksum for each of them and for every path in ``include`` that the
    caller wrote to ``out`` itself. Nothing else in the directory is adopted: a file
    the record does not name is not part of the record. A record is written once;
    ``overwrite`` is for fixture exporters that regenerate in place. Returns the dir."""
    out = Path(out)
    if (out / MANIFEST).exists() and not overwrite:
        raise FileExistsError(f"{out} already holds a record")
    out.mkdir(parents=True, exist_ok=True)
    for rel, obj in (files or {}).items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, (bytes, bytearray)):
            path.write_bytes(obj)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(jsonable(obj), f, separators=(",", ":"))
    named = sorted({Path(rel).as_posix() for rel in (*(files or {}), *include)})
    manifest = {**manifest, "checksums": {rel: sha256_file(out / rel) for rel in named}}
    if validate is not None:
        validate(manifest)
    with open(out / MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))
    return out


CACHE_SUFFIXES = (".npz",)


def read(path: PathLike) -> dict:
    """Load a record's manifest; raises if the address no longer matches its
    ``config_hash`` or a named file changed or is missing. Raw arrays (``.npz``)
    are caches, gitignored and re-derivable, so a missing one is allowed; a
    present one must still match."""
    out = Path(path)
    with open(out / MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    if "address" in m and m.get("config_hash") != address_hash(m):
        raise ValueError(f"{out / MANIFEST}: address block does not match config_hash")
    for rel, digest in m.get("checksums", {}).items():
        if not (out / rel).is_file():
            if rel.endswith(CACHE_SUFFIXES):
                continue
            raise FileNotFoundError(f"{out / rel} listed in manifest but missing")
        if sha256_file(out / rel) != digest:
            raise ValueError(f"{out / rel} does not match its manifest checksum")
    return m


def index(root: PathLike) -> List[dict]:
    """Manifests of every record under ``root``, newest first, each with a ``path`` key."""
    root = Path(root)
    found = []
    for mpath in root.rglob(MANIFEST):
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        found.append({**m, "path": mpath.parent.relative_to(root).as_posix()})
    return sorted(found, key=lambda m: m.get("created", ""), reverse=True)
