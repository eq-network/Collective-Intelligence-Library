# Changelogs

Paper trail of architectural changes to Mycorrhiza. Each file is named `{date}_{what-changed}.md`.

## 2026-03-30: Core Primitives (Plan 1 Phases A-C)

| Changelog | File Changed | Summary |
|-----------|-------------|---------|
| [pytree-fix](2026-03-30_pytree-fix.md) | `core/graph.py` | Partition global_attrs into dynamic (JAX arrays) and static (Python scalars) for JIT |
| [transform-decorator](2026-03-30_transform-decorator.md) | `core/category.py` | `@transform(reads, writes)` metadata + compose() propagation + bug fixes |
| [pipeline-compiler](2026-03-30_pipeline-compiler.md) | `core/pipeline.py` | Derive execution order from read/write DAG, topological batching |
| [schedule-primitive](2026-03-30_schedule-primitive.md) | `core/schedule.py` | Cadence + phase offset, `lax.cond` for JIT, schedule as experimental variable |
| [composition-operators](2026-03-30_composition-operators.md) | `core/category.py` | `parallel()` (disjoint writes merge) + `conditional()` (predicate-gated) |
| [fishing-commons-state](2026-03-30_fishing-commons-state.md) | `experiments/fishing_commons/` | State factory + type contracts for market/network/democracy |

## Visual Reference

- **[2026-03-30_core-primitives.pdf](2026-03-30_core-primitives.pdf)** — TikZ diagrams: pytree partition, typed transforms, dependency DAG, schedule timeline, composition operators, type contracts, state structure
- Source: [2026-03-30_core-primitives.tex](2026-03-30_core-primitives.tex) (recompile with `pdflatex`)
