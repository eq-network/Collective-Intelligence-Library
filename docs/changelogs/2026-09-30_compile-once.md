# 2026-09-30: compile once per spec

**Files changed:** `src/cilib/environments/spec.py`, `src/cilib/core/scan.py`,
`src/cilib/core/reduce.py`, root `conftest.py` (+ `environments/tests/test_spec_jit.py`)

## What

`EnvSpec.run`, `run_batch`, `run_reduced`, `run_reduced_batch` compile their scan once per
spec and call signature and keep the program on the spec. Every result is bit-identical to
the eager path they replace. The test suite turns on JAX's persistent compile cache.
No environment, mechanism, or experiment changed.

## Why

The perf baseline (`experiments/perf_baseline`, 2026-09-10) measured that the eager scan
recompiled on every call: 0.3 to 0.9 s per call against a 1 to 6 ms run. After this change,
a repeat call costs 3 to 15 ms (`run`) and 14 to 73 ms (8-seed `run_batch`); the full test
suite went from 557 s to 200 s with a warm cache.

## Bit-identity, and the three things that break it

A plain `jax.jit` around `run` changed float32 results in 10 of 13 environments. Three
separate causes, each handled in `EnvSpec._compiled` / `_init_batch`:

1. **A compiled `init_fn` rounds differently** (9 of 13): XLA fuses the initial-state
   arithmetic. Init (and reducer `init` / `finalize`) stay eager, outside the program.
2. **Fields `vmap` leaves unbatched.** The eager `run_batch` scanned seed-constant fields
   (a fixed network, a starting stock) unbatched. Batching them compiles a different
   program. Each field's batching is recorded eagerly and passed on as `in_axes`.
3. **Closed-over arrays become constants.** Arrays a `round_fn` captures are baked into a
   jitted program, and XLA constant-folds math on them with the host's libm. The eager scan
   received them as runtime arguments; so does the compiled path now, along with the step
   indices. No registered default config hits this; the red-team pass found it with a
   model where a plain jit differs in 1391 elements.

`tests/test_spec_jit.py` asserts exact equality for every registered environment, the sparse
adjacency, and a model built to trigger each cause.

## History: the 2026-09-11 attempt

Branch `jit-boundary` (927f99e, merged unchanged into history on 2026-09-30) made the same
move three weeks earlier and was never merged. It found causes 1 and 2 but made the
opposite call on 2: it batched every field, so one seed of `run_batch` equals `run` for the
same key, at the cost of moving past batch results in five environments (`compute_economy`,
`io_economy`, `task_economy`, `coupled_society`, `ledger_society`). In `ledger_society` the
rounding difference crossed an enforcement threshold and rewired the delegation matrix
within 12 steps. It did not see cause 3. Its coverage of the reduced paths is kept here.

## Consequence

- **No reported number moves.** Everything run before this date reproduces bit for bit.
- **`run` and one seed of `run_batch` still differ**, as they always did: in 6 of 11
  default environments, by up to 0.03 in `ledger_society` at T=50. Whether they should be
  made equal is an open decision in `ROADMAP.md`.
- **Closed-over Python state is read once.** A `round_fn` that reads a mutable dial keeps
  the value from the first call; configs are frozen for this reason.
- **Reuse the spec.** A fresh `make_env` or a fresh `trace_fn` lambda per call is a fresh
  compile.
