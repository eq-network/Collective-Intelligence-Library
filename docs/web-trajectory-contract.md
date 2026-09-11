# Web trajectory contract — the boundary between the engine and the lab page

Status: **v1.1, in use** (2026-07-24). Consumed by the eq-network playground
(`eq-network/prototypes/playground.html`, later `src/components/lab/sim/`).
Producer: `examples/05_export_trajectory.py`. Change this file and both sides
together or not at all. Since 2026-09-11 the producer writes the payload as
`trajectory.json` inside a `cilib.runs` record directory (manifest beside it);
the payload schema is unchanged.

v1.1 (2026-07-24): `adj` is no longer reserved — `value_contagion` (A3/C2) is
the first network game and emits its static `friendship` matrix; new optional
`system` field carries the pipeline DAG derived from `@transform` metadata
(`environments/system_graph.py`).

## Why

The lab page's playground replays runs of this library's environments in the
browser. The page must not care *where* a run came from — an in-browser JS port
of the dynamics (instant sliders), a JAX HTTP endpoint (heavy games, later), or
a file exported from the real engine (published-run replays, parity fixtures).
One JSON schema is the wire format for all three, so "should this game compute
client-side or server-side?" stays a per-game deployment choice, never a
rewrite. It is the JS shadow of contracts already frozen here: the trace dict
(`EnvSpec.trace_fn`, stacked by `run_scan`) and the GameSpec boundary
([game-boundary-design.md](game-boundary-design.md)).

## The schema

One JSON object per rollout:

```jsonc
{
  "meta": {
    "gameId": "governed_commons",   // env REGISTRY name (web may alias, e.g. "commons")
    "T": 500,                        // steps
    "N": 20,                         // agents
    "seed": 0,
    "params": { "K_cap": 500.0, "growth_rate": 0.35, "condition": "baseline", ... },
    "scalars": { "stock_pct": 0.0, "influence_fidelity": 0.001, ... }   // final metrics
  },
  "global": { "resource_level": [/* T floats */], "policy_target": [ ... ] },
  "node":   { "harvest": [/* T*N floats, row-major t*N+i */], ... },
  "static": { "principal_pref": [/* N floats */], "alignment": [ ... ] },
  "adj":    { "friendship": [/* N*N, row-major */] },   // network games; static per
                                                        // run (from finals) or [T*N*N]
  "system": {   // optional; the pipeline DAG as communication — derived from
                // @transform reads/writes by environments/system_graph.py, so the
                // System view / future graph editor gets a topology nobody hand-drew
    "nodes": [
      { "id": "culture", "kind": "field", "family": "node_attrs", "shape": [40] },
      { "id": "rng_key", "kind": "field", "family": "global_attrs", "shape": [2],
        "bookkeeping": true },              // plumbing: render dimmed, never hidden
      { "id": "adopt", "kind": "transform", "reads": ["broadcast_effort", ...],
        "writes": ["culture", "rng_key"] }
    ],
    "edges": [ { "from": "friendship", "to": "adopt" }, { "from": "adopt", "to": "culture" } ]
  }
}
```

Consumer note for `system` (live since 2026-07-27): the playground consumes it
— `Trajectories.fromJSON` passes it through verbatim, and a generic
`PipelineScene` (one renderer, zero per-game code) draws any game's DAG as a
transform spine with read-dominant fields above and write-dominant below.
Consumer-side conventions layered on top, not producer fields: (a) the three
JS-ported games embed exporter-generated *fixtures* of their fully-defended
graphs and filter them by mechanism toggle (substring match on the transform
id — schedules wrap the mechanism name), mirroring additive attachment; a
payload-carried `system` overrides the fixture. (b) surviving mechanism
transforms get a `color` annotation stamped consumer-side; producers should
not emit `color`. (c) fields with no edges render as shelved "outside the
declared pipeline" — the GameSpec boundary (observe → policy → actions)
carries no transform metadata and appears via the action fields in `node`, so
policy-read fields like `principal_pref` are honestly edge-less here.

The browser lifts each array into a `Float64Array` and indexes `node` fields as
`series[t * N + i]`. No nesting, no per-step objects — flat arrays keep a
500-step payload small and the consumer allocation-free.

## Mapping from a trace

Mechanical, no per-environment code:

- trace field of shape `(T,)` → `global`
- trace field of shape `(T, N)`, varying over time → `node` (flattened row-major)
- trace field of shape `(T, N)`, constant over time → `static` (first row) —
  a static per-agent draw reaches the payload by riding along in the trace
  (e.g. `principal_pref` today). A scene that needs a draw the trace omits
  (e.g. `alignment` in governed_commons) needs it added to that env's
  `default_trace` first — one line, and this mapping picks it up unchanged.
- `meta.params` = `dataclasses.asdict(env.config)` + the condition name
- `meta.scalars` = `env.evaluate(trace)` (single-run metric suite), floats

## Producer obligations

1. **Exact field names from the trace.** The web side keys charts and scenes on
   trace names (`resource_level`, `policy_target`, …); renaming a trace field is
   a breaking change to the page and must be flagged in a PR touching this file.
2. **Post-pipeline timing.** Values at index `t` are the state *after* tick
   `t`'s full pipeline (substrate → mechanisms → counter), i.e. exactly what
   `default_trace` records. The in-browser ports mirror this timing.
3. **Determinism.** `(gameId, params, seed) → identical payload`. Seeds are
   producer-local: JAX Threefry and the browser's mulberry32 streams never
   match bit-for-bit. Cross-producer comparisons are *qualitative* (collapse
   timing, plateau levels, metric ordering) — the validation ladder, not
   bit-exactness, is the fidelity gate on the JS side.

## Consumers

- `StaticEngine` (web): fetches a payload verbatim.
- `RemoteEngine` (web, future): a JAX endpoint returns this same object per
  POST `{gameId, params, seed}`.
- Parity eyeballing: export at the browser's defaults and compare trajectory
  shape against the in-browser run (`prototypes/playground.html`, seed shown
  in its UI).

## Producers (additive note, 2026-07-30 — no version change)

The generic trace→payload mapping lives in `cilib.environments.webexport`
(`trajectory_payload`); `examples/05_export_trajectory.py` and the sweep-bundle
exporter (`experiments/gd_bundles/`, per `docs/dial-lattice-design.md`) both emit
through it. Two producer options, both within v1.1:

- A producer MAY append **derived (T,) series** (engine-side reductions of the
  trace, e.g. `human_income_share`) as additional `global` fields — rule 1's
  "field names are engine trace names" extends to these; they are still
  engine-computed values, never view-side math.
- A producer MAY omit `adj` (it is already optional) and MAY whitelist trace
  fields; whatever ships must obey rules 1–3 unchanged.
- A trace field MAY be an **integer index series** (e.g. `top_listen_target`,
  per-node argmax neighbor ids) — serialized as ints, exempt from rounding.
  Consumers treat index values as ids valid at their own tick only and MUST NOT
  interpolate them across ticks.
