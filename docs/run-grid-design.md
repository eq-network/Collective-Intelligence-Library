# run_grid: dial grids as runtime data

**Status:** built and red-teamed 2026-10-06 on branch `run-grid`. Engine only; no
environment or experiment uses it yet.

## The problem

A sweep runs one model at many settings of a few dials, many seeds each. Seeds were
already cheap, since `run_batch` puts them under one `vmap`. Dials were not. Every dial
was config that the transforms close over, so each grid cell needed its own `EnvSpec`
and paid its own compile. The perf baseline measured a compile at 0.3 to 0.9 s against
1 to 6 ms to run 200 steps, in `experiments/perf_baseline/README.md`. Every sweep
driver in the repo pays this: `gd_bundles`, the WP1 to WP3 runners, the schedule scan.
Its 125-cell `ledger_society` bundle took 232 s, 1.86 s per cell, on the machine below.

## The idea

A dial value that reaches the compiled program as an input, and is not baked into it as
a constant, can change without a recompile. `GraphState` already allows this: an array
entry of `global_attrs` is a traced pytree child, while a Python scalar there is static
aux. `ledger_society` already reads six dials this way, the `*_now` globals that
`state.py` seeds from config.

`EnvSpec.run_grid` builds on that:

```
run_grid(key, n_seeds, n_steps, grid, reducers, trace_fn=None, batch=1)
  grid = { global_attrs key : array (G, *entry_shape) }
  -> (final_states, traces, reduced), leading axes (G, n_seeds)
```

Cell `g` runs every seed with `global_attrs[k] = grid[k][g]`. The pure helper
`with_globals` swaps the value in after `init_fn` and before the scan. The program is
the `run_reduced_batch` program wrapped in a `lax.map` over cells. Its compile-cache key
holds the shapes and dtypes of the grid, never its values, so a new set of values of
the same shape reuses the program.

Every cell uses the same key. Seed `s` starts from the same state and draws the same
random numbers in every cell, so two cells are compared as paired differences over
seeds. `lab/analysis/bootstrap.paired_diff_ci` does that comparison. Its interval
covers seed variation only; a reading across hundreds of cells needs its own care
about multiple comparisons.

## Invariants

Each one has a test in `src/cilib/environments/tests/test_run_grid.py`.

1. **Lifting a dial changes no number at its default, or it is opt-in.** A model that
   reads a dial from `global_attrs` should be bit-identical, on `run`, `run_batch` and
   `run_reduced_batch`, to the same model with the dial closed over, when the global
   is seeded with the config value. This has to be checked per dial; see "Lifting can
   move numbers" for a case where it failed and what was done.
2. **One program per grid shape.** New grid values add no program. A new cell count,
   seed count, run length, reducer set or `batch` adds one.
3. **With `batch=1`, a cell is bit-identical to `run_reduced_batch`** on a spec whose
   config holds the cell's value: the trajectory, the final state, and every reducer
   value the scan writes. Tested on a toy model and on `ledger_society` for
   `churn_now` and `repair_rate_now`. The one exception is below.
4. **Only array globals are dials.** A grid key that is absent from the initial
   `global_attrs`, is a Python scalar there, is weakly typed, or is `step` or `rng_key`
   raises `ValueError`. Grid values are cast to the entry's dtype, and a value that
   the cast would change, such as 2.9 for an int32 onset, raises.

**The exception to invariant 3.** Reducer accumulators are built inside the program,
from the state after the override, so an `init` that reads a dial sees the cell's
value. The eager reference builds them op by op. Inside a program XLA folds constant
subexpressions and rewrites some divisions, so a readout like an entropy divided by
`log(N)` can come out one float32 ulp, the last binary digit, away. This matters only
where the scan never overwrites the init value: `at_step` at onset 0 or past the end of
the run, and `last` at T = 0. The gap stays in that output and never feeds back into
the dynamics.

## Why cells run one at a time by default

The first version put the cells under a `vmap`. That batches every quantity the dial
touches, and XLA can then sum in a different order. The `run` versus `run_batch` gap is
the same effect, and in `ledger_society` it once crossed an enforcement threshold and
rewired the delegation matrix within 12 steps, per
`docs/changelogs/2026-09-30_compile-once.md`.

A `lax.map` with one cell at a time runs each cell through the `run_reduced_batch`
program, so the numbers do not move. `batch > 1` vectorizes that many cells for speed.
Measured on `ledger_society` at its defaults: 25 cells over `churn` and `repair_rate`,
8 seeds, T = 400, late-window human income, attention and power shares, compared with
a Python loop of `make_env` cells.

| path | first call | repeat call | cells equal to the loop |
|---|---|---|---|
| loop of `make_env` + `run_reduced_batch` | 44.7 s, 25 compiles | | reference |
| `run_grid`, `batch=1` | 3.5 s | 1.5 s | all 25, bit for bit |
| `run_grid`, `batch=25` | 2.3 s | 0.6 s | 18 of 25 |

Provenance: jax 0.9.0, CPU, Intel 64 family 6 model 141, 2026-10-06,
`python -m experiments.perf_baseline.grid_calls`. One run per row; a red-team rerun on
the same machine gave 46.9 s, 2.8 s and 2.0 s. The timings are indicative and carry
no interval. The equality column is exact.

On this grid the `batch=25` cells that differed did so only at float32 rounding. That
describes this grid and run length; it is not a bound, and a model with thresholds can
amplify such a gap. Read `batch > 1` results only against each other.

## Lifting can move numbers

Invariant 1 is a claim to check per dial, not a property of the method. Lifting
`ledger_society`'s three channel dials on 2026-10-06 was checked against a snapshot
of the old code: five configs, from the defaults to every channel at its maximum
with no repair, at T = 400, on `run`, `run_batch` and `run_reduced_batch`, every
field compared bit for bit.

- `run`, one seed: unchanged in all five configs.
- The seed-batched paths: `attention_to_ballots` lifted alone changed nothing.
  `reach_per_spend` or `regime_rate` lifted alone each changed the batched numbers.
  At the defaults the change was near float32 rounding, about 1e-7. In the
  maximum-coupling config it grew to 0.003 in the per-tick human power share,
  because the run carries it through thresholds for 400 steps.

The cause is that XLA compiles a constant and a runtime input differently once
seeds are under `vmap`; the arithmetic is the same. Changing past batch numbers
was not acceptable, so the three dials are lifted only under the structural flag
`traced_channels`. Off, the default, the code is the old expression, and the
snapshot check found all 798 compared arrays unchanged. On, they are `run_grid`
dials. A sweep that turns the flag on must say so, and its numbers are not
bit-comparable with results made with it off.

The lesson for the next dial: lift it, compare a before/after snapshot of the
batched paths, and put it behind a flag if anything moved.

## What it does not do

- **Dials that shape the initial state.** `init_fn` runs once, with the config value.
  The override reaches `round_fn` only.
- **Globals a mechanism writes.** The override sets a global's value in the initial
  state. If a lever rewrites it each tick, the grid value lasts one round:
  `reach_cut_now` under `policy_levers`, and `gamma_w_now`, `update_rate_w_now` and
  `reach_cut_now` under the culture levers. Sweep the lever's plan instead. The same
  holds for state variables such as a resource stock: the grid sets the starting value.
- **Structural config.** Population size, horizons and which mechanisms exist change
  the program and stay closed over. One `run_grid` call per structural setting.
- **Comparisons with older sweeps.** `gd_bundles` and the WP runners score traces from
  `run_batch`. `run_grid` cells equal `run_reduced_batch`, whose reducers match trace
  metrics to float32 rounding only, per `metrics/reducers.py`. A `run_grid` number is
  not a reproduction of a `gd_bundles` number.
- **Fresh reducers per call.** The cache is keyed on the reducer and `trace_fn`
  objects. A driver that rebuilds them each call, through `make_reducers(cfg)` or an
  inline `late_mean(f)`, recompiles every call. Build them once.

## Pieces added with it

- **Onset from state.** `scheduled(mechanism, onset="t_on_now")` reads the switch-on
  step from a `global_attrs` entry each tick, so it can be a grid dial. An int onset
  runs the old code unchanged. `ScheduleSpec.onset` takes either, and a string onset
  round-trips through `cilib.runs` records.
- **`at_step(readout, step_key)`** in `metrics/reducers.py`: the readout on the state
  just before step `global_attrs[step_key]`, the last state a mechanism switching on
  at that step has not touched. Its `init` is the readout of the initial state, so an
  onset of 0 is defined; this departs from the identity-init rule in `core/reduce.py`,
  and the docstring says so. It uses the scan index, which equals the `step` global
  when a run starts at step 0. A caller who also sweeps a plan must derive the onset
  dial from that plan; nothing ties the two together.
- **`paired_diff_ci(a, b)`** in `lab/analysis/bootstrap.py`: percentile bootstrap of
  `mean(a - b)` over paired seeds, plus the fraction of seeds whose own difference has
  the sign of the mean. It refuses 2-D input, which would pool cells, and NaN.

## Red-team record

Three independent attackers on 2026-10-06: equivalence, silent wrong results, and
evidence. Attacks and answers:

- **Bit-identity breaks at reducer init.** It holds. Answered by narrowing invariant 3
  to the exception above, with a test that pins it, rather than by a second path that
  builds accumulators eagerly per cell; that would add an accumulator batching layout
  to track, for a one-ulp gap in an output that feeds nothing.
- **`step` and `rng_key` were accepted as grid keys.** `rng_key` did nothing and `step`
  shifted the clock. Both are now refused.
- **Float values for an int dial were truncated** without warning. A value the cast
  would change now raises.
- **Lever-written globals are nearly inert as grid keys.** It holds and cannot be
  detected cheaply; recorded under "What it does not do".
- **Tests passed on broken code.** Removing `batch` from the cache key, removing the
  dtype cast, or building accumulators before the override all left the suite green.
  Each now has a test that fails on it.
- **Rebuilt reducers recompile.** It holds for `run_reduced_batch` too; documented.
- **`frac_same_sign` was wrong at a zero mean** and accepted NaN and 2-D input. It is
  NaN at a zero mean now, and the other inputs raise.
- **The onset dial and the plan are independent,** and a plan horizon shorter than the
  run repeats its last row. Both concern the experiment that builds the plan, not the
  engine; recorded here for it.
- **Held where the attackers looked:** bit-identity of the dynamics at T = 400 for five
  `ledger_society` dials, `policy_plan`, and one dial each in `capital_economy`,
  `governed_commons` and `task_economy`; `at_step` against scheduled firing at onsets
  0, 5, 10, 29 and 30; shared seeds across cells; the bootstrap itself.

## Open

- The `batch > 1` gap was measured on one grid. Whether a longer run or another model
  turns it into a visible difference was not measured.
- When `run_grid` sits inside a caller's `jit`, it builds its program from the eager
  functions. In the tests that matches the direct path within float32 rounding, not
  bit for bit.
