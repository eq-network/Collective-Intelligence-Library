# perf_baseline — where the time goes

The measured starting point for any engine optimisation: compile against run,
eager against jitted, seed batches, and population scaling, for every
registered environment at its default configuration. Nothing here changes the
engine; it says what would be worth changing. Numbers are machine-specific and
travel only with the provenance line above each table.

```bash
python -m experiments.perf_baseline.run --scaling      # -> baseline.json (all envs)
python -m experiments.perf_baseline.run --envs ledger_society influence_exchange \
    delegative_polity capital_economy value_contagion --scaling --mults 1 8 32 64 \
    --seeds 1 32 --out experiments/perf_baseline/scaling-large.json
python -m experiments.perf_baseline.cache_probe [cache_dir]   # run three times, see docstring
python -m experiments.perf_baseline.repeat_calls              # compile-once, after 2026-09-30
python -m experiments.perf_baseline.table [file.json]         # the tables below
```

`baseline.json` and `scaling-large.json` are committed: scalars plus provenance,
a few kilobytes. Re-run on a different machine and the tables regenerate from
the new file; do not edit them by hand.

## Findings

**1. Compile costs two orders of magnitude more than running.** At every
environment's default size, compiling the scan takes 0.3 to 0.9 s and running
200 steps takes 1 to 6 ms. Compile is a fixed cost: the estimate at T=50 and at
T=200 agree within noise for every environment, as expected for a `lax.scan`
whose body is traced once.

**2. The eager path recompiles on every call.** `EnvSpec.run` and `run_batch`
call `lax.scan` and `vmap` with no `jax.jit` around them, and a second eager
call costs the same as the first (the `eager call 2` column equals `jit call 1`,
not `jit warm`). This is how every test and every study calls the engine
today, so a loop of K runs pays K compiles. JAX's persistent compile cache
(`jax_compilation_cache_dir`, off by default) takes a repeated eager call on
`ledger_society` from 0.60 s to 0.19 s, the remainder being the re-trace and the
cache round trip; the same call under `jax.jit` repeats in 4 ms. Measured with
`cache_probe.py`, same machine and revision as the tables.

**3. Seeds are close to free.** Batching 32 seeds under `vmap` costs 2 to 8
times a single seed, not 32 times: efficiency of 200 to 800 percent in the
seed table. A single-seed run at these sizes is dominated by per-step
overhead inside the scan, and the batch amortises it. The marginal seed at
S=32 costs about 2 ms per 200 steps on the heaviest environment. There is no
reason to run fewer than a few dozen seeds.

**4. Dense adjacency is the ceiling, and it is closer than the small sweep
suggests.** Below roughly 300 nodes every environment sits on the overhead
floor, so the exponents in the first scaling table are floor artifacts and not
algorithmic (examples/07 makes the same warning). The large sweep clears the
floor. Over its whole range the three environments carrying dense `(N, N)`
ledgers grow at about N^1.6 to N^1.8, but the last doubling is far steeper:
`ledger_society` 326 ms to 2653 ms (N=646 to 1286), `influence_exchange` 353 ms
to 2634 ms and `delegative_polity` 363 ms to 3983 ms (N=964 to 1924), 7 to 11x
for 2x nodes. That is steeper than the N^2 matvec alone. The plausible cause is
the dense float32 matrices (6.6 MB each at N=1286, two of them in
`ledger_society`) leaving the CPU cache; that was not measured and is the first
thing to check. `capital_economy` shows the same jump (110 ms to 992 ms for
N=652 to 1292) although its only graph is a 6-sector recipe: the technical
matrix is stored padded to `(N, N)` with an `(S, S)` live block, and
`distribute` rebuilds an `(N, N)` sector one-hot every tick, so each tick pays
three dense `N x N` products (`rebalance`'s `A @ x`, `onehot.T @ K_own`,
`onehot @ share_priv`) over matrices with at most `N x S` non-zeros. The same
storage is used by `io_economy` and `task_economy`. Extrapolated naively, the
Track 06 milestone of
10,000 agents costs minutes per 200 steps on this CPU and 400 MB per dense
float32 adjacency. This is the linear-algebra target: the adjacency ledgers
and the row-stochastic attachment kernel, not the scan.

## What this implies

- **Cheapest win: `jit` at the `EnvSpec` boundary.** `run` and `run_batch` with
  `n_steps` static. In-process repeats drop from a compile to a warm run, about
  100x at default sizes, with no model change. The equivalence is a
  bit-identity test, since an eager scan and a jitted scan can round differently
  once XLA fuses across the step (the postmortem's sealing finding).
- **Persistent compile cache for the test suite.** One config line, about 3x
  on cold repeats of identical programs (measured on one environment, not yet
  on the suite). It does not remove the re-trace. Jit at the boundary helps the
  suite less than it helps studies, because most tests build a fresh
  environment per call and a fresh closure is a fresh compile either way.
- **The structured case first: the economy environments' `(N, N)` recipe.**
  Index the sector block and build the one-hot as `(N, S)` instead; three
  quadratic products per tick become `N x S`, and the equivalence is a
  bit-identity test against the current pipeline.
- **The real work is sparse or structured adjacency for the ledger
  environments.** `value_contagion` already has the BCOO path and the
  equivalence test; `attachment.py` is the shared kernel. Until that lands, the
  10,000-agent milestone is blocked on memory and the N^2 matvec, not on
  compile or dispatch.
- **The trajectory is still the memory wall at scale.** At these sizes the
  default trace is under 1 MB; at 10,000 agents and T=2000 it is the O(T x N)
  object CLAUDE.md names, and the reducer path exists for it.

## Done since: compile once per spec, compile cache for tests (2026-09-30)

The first two moves above, landed on top of `b7f2d2d`. Same machine, jax 0.9.0,
CPU, T=200, persistent cache off unless stated.

- **`EnvSpec.run` / `run_batch` compile once per spec**
  (`python -m experiments.perf_baseline.repeat_calls`, all 13 environments). A
  repeat call with a new key went from 0.19 to 0.86 s (eager, recompiling) to 3
  to 15 ms; a repeat 8-seed batch from 0.25 to 1.35 s to 14 to 73 ms. A first
  `run` costs what an eager call did; a first `run_batch` costs 0.94 to 1.57
  times one, the price of tracing the program once more to lift its constants
  out (below). A loop of K calls on one spec breaks even at K = 2.
- **Bit-identical, by construction and by test.** A plain `jax.jit` around
  `run` was not: it changed float32 bits in 10 of 13 environments, from three
  separate causes, each now avoided in `EnvSpec._compiled`: (1) a compiled
  `init_fn` fuses and rounds differently (9 of 13), so init stays eager;
  (2) state fields `vmap` leaves unbatched were batched, which compiles a
  different program, so each field's batching is carried over; (3) arrays a
  `round_fn` closes over became constants that XLA folds with host libm, so
  they are passed as runtime arguments as the eager scan did. (3) does not
  occur in any registered default config and was found by the red-team pass;
  `tests/test_spec_jit.py` holds a model where it does (a plain jit differs in
  1391 elements, the compiled path in 0).
- **Persistent compile cache for the test suite** (root `conftest.py`,
  `.jax_cache/`, gitignored, about 60 MB after two runs). Full suite, 413
  tests: 557 s before either change (379 tests), 552 s with compile-once
  alone, 268 s cold and 200 s warm with both. Compile-once barely moves the
  suite, as predicted above: most tests build a fresh environment.
- **`run_reduced` / `run_reduced_batch`** go through the same compiled path,
  with reducer `init` / `finalize` kept eager for the same reason as `init_fn`.

## What we do not know yet

- **Why one full test run took 1 h 42 min.** It did not reproduce: the same
  command, same revision, later the same day took 6 min 21 s for 353 tests
  (`test-durations-2026-09-10.txt`). Call time totals 361 s over the 296 tests
  pytest timed, about 1.2 s per test, which is one to three compiles each: the
  suite is compile-bound, as finding 2 predicts, and the slow reading was a
  property of the machine during that hour (power management or other load,
  not recorded), not of the suite. The two heaviest items are the
  `08_streaming_metrics` example smoke test (51 s on its own) and the
  `ledger_society` ladder (47 s over 21 tests); together a third of the total.
- **Whether a single-seed run uses more than one core.** The super-linear seed
  efficiency is consistent with a single seed leaving most of the CPU idle.
  Thread utilisation was not measured.
- **Nothing here is a GPU number.** The backend is CPU throughout.

## Tables

### Per environment

Measured 2026-09-10 at `09b486f`, jax 0.9.0, TFRT_CPU_0, 12 cores, T=200, best of 3.

| env | N | eager call 1 | eager call 2 | jit call 1 | jit warm | compile | us/step | trace MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `capital_economy` | 32 | 0.60 s | 0.46 s | 0.58 s | 2.8 ms | 0.57 s | 14 | 0.21 |
| `commons_harvest` | 16 | 0.49 s | 0.43 s | 0.51 s | 6.5 ms | 0.50 s | 33 | 0.03 |
| `compute_economy` | 26 | 0.27 s | 0.25 s | 0.30 s | 1.6 ms | 0.30 s | 8 | 0.09 |
| `coupled_society` | 26 | 0.44 s | 0.43 s | 0.71 s | 2.7 ms | 0.70 s | 14 | 0.15 |
| `delegative_polity` | 34 | 0.30 s | 0.29 s | 0.47 s | 2.5 ms | 0.47 s | 13 | 0.11 |
| `governed_commons` | 20 | 0.29 s | 0.27 s | 0.33 s | 2.0 ms | 0.32 s | 10 | 0.07 |
| `influence_exchange` | 34 | 0.20 s | 0.18 s | 0.39 s | 1.7 ms | 0.39 s | 9 | 0.14 |
| `io_economy` | 22 | 0.47 s | 0.44 s | 0.59 s | 2.4 ms | 0.59 s | 12 | 0.49 |
| `ledger_society` | 26 | 0.64 s | 0.63 s | 0.88 s | 4.6 ms | 0.88 s | 23 | 0.32 |
| `task_economy` | 21 | 0.31 s | 0.28 s | 0.33 s | 1.6 ms | 0.33 s | 8 | 0.04 |
| `value_contagion` | 40 | 0.21 s | 0.20 s | 0.30 s | 1.3 ms | 0.30 s | 7 | 0.06 |

### Seed batches (jit warm; efficiency = S x single-seed time / batch time)

| env | S=1 | S=8 | S=32 |
|---|---:|---:|---:|
| `capital_economy` | 3 ms (100%) | 16 ms (134%) | 43 ms (199%) |
| `commons_harvest` | 5 ms (100%) | 22 ms (184%) | 58 ms (275%) |
| `compute_economy` | 2 ms (100%) | 3 ms (418%) | 8 ms (687%) |
| `coupled_society` | 3 ms (100%) | 10 ms (241%) | 21 ms (442%) |
| `delegative_polity` | 3 ms (100%) | 15 ms (151%) | 44 ms (201%) |
| `governed_commons` | 2 ms (100%) | 5 ms (365%) | 8 ms (829%) |
| `influence_exchange` | 2 ms (100%) | 7 ms (202%) | 21 ms (272%) |
| `io_economy` | 3 ms (100%) | 11 ms (186%) | 32 ms (253%) |
| `ledger_society` | 5 ms (100%) | 27 ms (145%) | 68 ms (225%) |
| `task_economy` | 2 ms (100%) | 4 ms (338%) | 7 ms (660%) |
| `value_contagion` | 1 ms (100%) | 3 ms (375%) | 6 ms (679%) |

### Population scaling, 1x to 8x (jit warm; p from the first and last column)

| env | 1x | 2x | 4x | 8x | p |
|---|---:|---:|---:|---:|---:|
| `capital_economy` | N=32: 3 ms | N=52: 3 ms | N=92: 6 ms | N=172: 15 ms | 1.01 |
| `commons_harvest` | N=16: 6 ms | N=32: 8 ms | N=64: 10 ms | N=128: 15 ms | 0.44 |
| `compute_economy` | N=26: 2 ms | N=46: 2 ms | N=86: 2 ms | N=166: 3 ms | 0.26 |
| `coupled_society` | N=26: 3 ms | N=46: 4 ms | N=86: 6 ms | N=166: 17 ms | 1.01 |
| `delegative_polity` | N=34: 3 ms | N=64: 5 ms | N=124: 15 ms | N=244: 38 ms | 1.35 |
| `governed_commons` | N=20: 2 ms | N=40: 2 ms | N=80: 2 ms | N=160: 4 ms | 0.28 |
| `influence_exchange` | N=34: 2 ms | N=64: 3 ms | N=124: 11 ms | N=244: 28 ms | 1.37 |
| `io_economy` | N=22: 2 ms | N=38: 3 ms | N=70: 5 ms | N=134: 9 ms | 0.74 |
| `ledger_society` | N=26: 5 ms | N=46: 9 ms | N=86: 17 ms | N=166: 42 ms | 1.19 |
| `task_economy` | N=21: 2 ms | N=41: 2 ms | N=81: 2 ms | N=161: 3 ms | 0.30 |
| `value_contagion` | N=40: 1 ms | N=80: 1 ms | N=160: 3 ms | N=320: 6 ms | 0.73 |

### Population scaling, 1x to 64x (`scaling-large.json`)

Measured 2026-09-10 at `09b486f`, jax 0.9.0, TFRT_CPU_0, 12 cores, T=200, best of 3.

| env | 1x | 8x | 32x | 64x | p |
|---|---:|---:|---:|---:|---:|
| `ledger_society` | N=26: 5 ms | N=166: 42 ms | N=646: 326 ms | N=1286: 2653 ms | 1.62 |
| `influence_exchange` | N=34: 2 ms | N=244: 28 ms | N=964: 353 ms | N=1924: 2634 ms | 1.82 |
| `delegative_polity` | N=34: 3 ms | N=244: 37 ms | N=964: 363 ms | N=1924: 3983 ms | 1.82 |
| `capital_economy` | N=32: 3 ms | N=172: 15 ms | N=652: 110 ms | N=1292: 992 ms | 1.60 |
| `value_contagion` | N=40: 1 ms | N=320: 6 ms | N=1280: 56 ms | N=2560: 787 ms | 1.53 |
