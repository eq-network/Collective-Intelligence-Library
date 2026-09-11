# Changelogs

Paper trail of architectural changes to the Collective Intelligence Library. Each file is
named `{date}_{what-changed}.md`. Historical note: entries predating 2026-06 describe the
pre-`src/` layout (`core/…`, `engine/…`) as it existed on their date.

## 2026-03-30: Core Primitives (Plan 1 Phases A-C)

| Changelog | File Changed | Summary |
|-----------|-------------|---------|
| [pytree-fix](2026-03-30_pytree-fix.md) | `core/graph.py` | Partition global_attrs into dynamic (JAX arrays) and static (Python scalars) for JIT |
| [transform-decorator](2026-03-30_transform-decorator.md) | `core/category.py` | `@transform(reads, writes)` metadata + compose() propagation + bug fixes |
| [pipeline-compiler](2026-03-30_pipeline-compiler.md) | `core/pipeline.py` | Derive execution order from read/write DAG, topological batching |
| [schedule-primitive](2026-03-30_schedule-primitive.md) | `core/schedule.py` | Cadence + phase offset, `lax.cond` for JIT, schedule as experimental variable |
| [composition-operators](2026-03-30_composition-operators.md) | `core/category.py` | `parallel()` (disjoint writes merge) + `conditional()` (predicate-gated) |
| [fishing-commons-state](2026-03-30_fishing-commons-state.md) | `experiments/fishing_commons/` | State factory + type contracts for market/network/democracy |

## 2026-07-24: From Core Primitives to a Benchmark Suite (April–July roll-up)

Covers everything between the two typeset documents (25 commits). Rather than one entry
per change, the design record for this period lives in `docs/`:

| Document | Covers |
|---|---|
| [game-boundary-design.md](../game-boundary-design.md) | `GameSpec` / `close()` — the open-environment boundary (**frozen**) |
| [abm-suite-design.md](../abm-suite-design.md) | Classical-ABM program: validation ladder, mechanisms-attach-via-scheduler |
| [alpha-plan.md](../alpha-plan.md) | Scenario → engine mapping, phasing, definitions of done |
| [model-register-design.md](../model-register-design.md) | The economy register: structural robustness, assumptions cards, forking |
| [cultural-register-design.md](../cultural-register-design.md) | The cultural register: separation × persuasion, spectral metrics |

## Visual Reference

- **[2026-07-24_alpha-benchmark-suite.pdf](2026-07-24_alpha-benchmark-suite.pdf)** — the
  April–July roll-up. TikZ diagrams: three-ring layout, the open-game boundary, schedule
  timeline, the two-counterfactual comparison, the economy register's substitutability
  bracket, task-frontier wage paths, the cultural phase diagram. Includes the
  `compute_economy` audit and the corrected benchmark numbers.
- **[2026-03-30_core-primitives.pdf](2026-03-30_core-primitives.pdf)** — TikZ diagrams: pytree partition, typed transforms, dependency DAG, schedule timeline, composition operators, type contracts, state structure
- Sources: the matching `.tex` files (recompile with `pdflatex <file>.tex`, twice for the
  table of contents)

## 2026-09-11: Compilation at the environment boundary

| Changelog | File Changed | Summary |
|-----------|-------------|---------|
| [jit-boundary](2026-09-11_jit-boundary.md) | `environments/spec.py` | `run*` compile once per program and are cached on the spec; init stays eager (bit-identity); the old `run_batch` unbatched seed-constant leaves, now fixed |
