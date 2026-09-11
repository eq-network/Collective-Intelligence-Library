# 2026-09-11: jit at the `EnvSpec` boundary

**File changed:** `src/cilib/environments/spec.py` (+ `environments/tests/test_spec_jit.py`)

## What

`EnvSpec.run`, `run_batch`, `run_reduced`, `run_reduced_batch` now build their scan under
`jax.jit` once per program and cache it on the spec, keyed by `(kind, n_steps, batched,
trace_fn, reducers)`. `init_fn` stays op by op outside the program; the initial state is
passed in as an argument. No environment, mechanism, or experiment changed.

## Why

The perf baseline (`experiments/perf_baseline`, 2026-09-10) measured that a bare `lax.scan`
recompiles on every call: 0.3 to 0.9 s per call against a 1 to 6 ms run. Every test and
every study called the engine that way. Measured on the same machine after this change,
T=200 at default size:

| environment | bare repeat call | `run` warm |
|---|---|---|
| governed_commons | 294 ms | 3.5 ms |
| ledger_society | 887 ms | 21 ms |
| capital_economy | 755 ms | 13 ms |

## Two findings, both from the bit-identity test

1. **`init_fn` under `jit` rounds differently in 8 of 11 environments** (up to 2e-6 absolute
   on normalized float32 fields): XLA fuses the initial-state arithmetic. The scan itself
   compiles bit-identically in every environment. So init runs eagerly and `run` is
   bit-identical to the pre-change path everywhere.
2. **The old `run_batch` was not the batch it claimed to be.** It traced init and scan under
   one `vmap`, so seed-constant init leaves entered the scan unbatched and were accumulated
   in a different order than batched operands. Five environments (`compute_economy`,
   `io_economy`, `task_economy`, `coupled_society`, `ledger_society`) moved at float32
   rounding, and in `ledger_society` a rounding difference crossed an enforcement threshold
   and rewired the delegation matrix within 12 steps. The old numbers are reproduced exactly
   by passing those leaves with `in_axes=None`, which is how the mechanism was confirmed.
   `run_batch` now passes every leaf with its seed axis, matching per-seed `run`.

## Consequence

Batch results committed before this date for those five environments may differ from a
rerun at the last float32 digit; where a claim rests on such a run, rerun it. The suite
keeps its behavioral tests (direction, ordering), which is why none of them pinned the old
digits.

## Deferred

- The test suite still pays a compile per test because each test builds a fresh
  environment; the persistent compile cache (perf card, move 2) is the next step for that.
- A fresh `trace_fn` lambda per call is a fresh program; the cache holds a reference to
  each, so a loop that makes a new lambda every call grows the cache. Pass one object.
