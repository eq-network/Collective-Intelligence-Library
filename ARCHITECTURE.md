# Architecture — the pattern map

CI Lib is a JAX-native framework for multi-agent simulation built on one idea:
**a simulation is a pure function over an immutable state, scanned across time.**
Everything else is a way to *compose* and *catalog* those functions.

This document is the map a reader (human or coding agent) should hold before
touching the code. For *how to add* a piece, see [EXTENDING.md](EXTENDING.md).

## The one sentence

```
GraphState  --Transform-->  GraphState  --Transform-->  ...        (one tick)
        run_scan folds this over T ticks;  vmap batches it over seeds
```

## The relational vocabulary

The framework is a small category-theory kernel. Each concept has exactly one job:

| Concept | What it is | In code |
|---|---|---|
| **GraphState** | the immutable state (a JAX pytree): `node_attrs`, `adj_matrices`, `edge_attrs`, `global_attrs` | `core/graph.py` |
| **Transform** | a morphism `GraphState -> GraphState` (a pure round step) | `core/category.py` |
| **`@transform`** | tags a Transform with its `.reads` / `.writes` (declared effects) | `core/category.py` |
| **`sequential`** | monadic **bind** — run transforms in order | `core/category.py` |
| **`identity`** | monadic **return** — the no-op transform | `core/category.py` |
| **`parallel`** | run on the same input, merge **disjoint** writes | `core/category.py` |
| **`compile_pipeline`** | **automatic Kleisli composition** — derives execution order from the read/write DAG, batches independent steps, rejects cycles & write-conflicts | `core/pipeline.py` |
| **`ScheduleEntry`** | fire a transform every *k* ticks (cadence as an experimental dial) | `core/schedule.py` |
| **`run_scan` / `run_scan_batch`** | the compiled runner: `lax.scan` over ticks, `vmap` over seeds | `core/scan.py` |

The shift from `sequential(vote, harvest, reward)` to
`compile_pipeline([vote, harvest, reward])` means **you declare what each step reads
and writes, and the compiler derives the order** — not the other way around.

## Two tiers of time

The boundary is a property of the *agent*, not the engine:

- **Pure tier** (`core/scan.py`): every agent + the world is side-effect-free →
  compiled `lax.scan`, `vmap`-able over seeds/dials as one program. The default;
  where research paradigms live.
- **Eager tier** (`core/time.py`): for genuinely effectful agents (live LLM / sensor
  calls). Slower, not `vmap`-able over the effect. Reach for it only when a node must
  call the outside world mid-loop. `GraphState` is the interface between tiers.

## The catalog model — swappable building blocks

The library ships **catalogs** of pre-made, interchangeable blocks. A catalog is a
registry of factories that share one *type function* — `Config -> TypedCallable` —
so any entry is swappable for a sibling. Contracts live in
[`core/protocols.py`](src/cilib/core/protocols.py); each catalog's `REGISTRY` is a
plain dict in its `__init__.py` (open it, see every entry).

| Catalog | Type function | Produces | Package |
|---|---|---|---|
| **agents** | `Config -> Policy` (or `PureAgent`) | a decision rule `(obs,key)->action` | `cilib/agents` |
| **transformations** | `Config -> Transform` | an atomic `state->state` step | `cilib/transformations` |
| **mechanisms** | `Config -> Transform` (typed, family-disjoint writes) | a composed institution (market / network / democracy) | `cilib/mechanisms` |
| **environments** | `(**cfg) -> EnvSpec` | a runnable substrate | `cilib/environments` |

Above the catalogs sit two more roles:

- **paradigms** (`cilib/paradigms`) — *composers*: a self-contained model (agents +
  world + dynamics) wiring catalog entries into a pipeline. A new study = a new
  folder following the 6-part contract in `paradigms/README.md`. Examples:
  `active_inference`, `polycentric`.
- **analysis** (`cilib/analysis`) & **metrics** (`cilib/metrics`) — *measurement*:
  analysis is `GraphState`-free offline math (effective information, causal
  emergence); metrics are in-loop composable readouts. The simulation tier produces
  trajectories; this tier measures them.

An **experiment** (`experiments/`) selects catalog entries, compiles a pipeline, and
sweeps it over seeds/dials.

## Layering & import DAG

Strictly one-directional — nothing below depends on anything above it:

```
core/                       framework + contracts (depends on nothing in the library)
  ▲
agents/ transformations/ mechanisms/ environments/      catalogs   (import core)
  ▲
paradigms/                  composers  (import core + catalogs)
  ▲
experiments/                studies    (import paradigms, analysis, metrics)

analysis/                   measurement — imports NOTHING from the library (offline)
```

## Where things live

```
src/cilib/        the installable library (import root: `cilib`)
  core/           GraphState, Transform, @transform, compile_pipeline, scan, schedule, protocols
  agents/  transformations/  mechanisms/  environments/     the catalogs
  paradigms/      active_inference, polycentric  (6-part contract)
  analysis/  metrics/   measurement
  execution/      experiment-runner infra
experiments/      in-repo studies (import cilib; not part of the package)
```
