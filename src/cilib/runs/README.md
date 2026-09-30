# runs — the run store

The only place the library touches the filesystem. A **record** is a directory:

```
<root>/<UTC stamp>_<name>/
  manifest.json      address + metric ids + engine + checksums   (committed)
  scorecard.json     or scalars.json, trajectory.json, ...       (committed, small)
  *.npz              raw arrays, a cache: re-derivable, ignored
```

The manifest is written last and names every file that belongs to the record
with its sha256. `read` re-derives the address hash and every checksum, so an
edited manifest, an edited file, or a missing file is an error; a missing `.npz`
is not, since it is a cache.

```python
from cilib import runs

m = runs.manifest(env="governed_commons", config=env.config, T=200, seed0=0, n_seeds=32,
                  kind="scorecard", conditions=scn.conditions, metrics=env.metrics)
out = runs.write(runs.run_dir("experiments/benchmark/runs", "commons", m["created"]), m,
                 files={"scorecard.json": scorecard})
runs.read(out)                              # manifest, verified
runs.index("experiments/benchmark/runs")    # every manifest under a root, newest first
```

**The address.** `manifest["address"]` is everything a replay needs: `env`,
`config`, `overrides`, `mechanisms` or `conditions` as `(catalog key, config,
schedule)` records, `T`, `seed0`, `n_seeds`, plus any `extra` design fields (a
sweep's axes). `config_hash` is the sha256 of that block canonically serialized,
so two records with the same hash describe the same experimental design. It is a
design hash, not a payload or software hash: a scorecard and a trajectory of the
same design share it, and a JAX upgrade does not change it. Beside it: `kind`
(what was recorded), `metrics` (id, kind, version, definition hash from
`cilib.metrics.identity`), `engine` (package version, the `source` directory that
was imported, git revision and repository-wide dirty flag, JAX version, backend,
x64), `created`, `notes`.

**One record is one design over its seeds.** A single rollout is `n_seeds=1`; a
comparison of conditions is one record with `conditions`; a sweep bundle is one
record per bundle. This resolves the open question in
`docs/observability-design.md`: the experiment is the primitive, not the seed.

**Only named files join a record.** `write` checksums the files it writes plus the
paths the caller lists in `include`; it never scans the directory, so a leftover
from an earlier run cannot become provenance. A record is written once;
`overwrite=True` exists for fixture exporters that regenerate in place.

**Who writes through it.** `experiments/benchmark/run.py` (scorecards, with the
two headline instruments among the metric ids), `examples/05_export_trajectory.py`
(trajectory records), and the two fixture exporters `experiments/gd_bundles` and
`experiments/library_explainer`, whose manifests spread their schema-bound fields
over this core so the view contract's layout is unchanged and the address block
rides along. `experiments/perf_baseline/run.py` records `engine_info` only: its
output is a flat, incrementally rewritten timing file, not an address.

**Known limits, stated.** The git revision is that of the imported `cilib`
package; under an editable install pointing at another checkout (a worktree), run
with `PYTHONPATH=src` or the `source` field will show the mismatch. Records from a
dirty tree carry `git_dirty: true` and prove nothing about the code.
`experiments/gd_game/export.py` is dead-end code and still hashes metrics its own
way. Not a database: if `index` ever feels slow, look at the run count.
