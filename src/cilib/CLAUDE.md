# Engine conventions (src/cilib)

Load-bearing rules for all engine code. The root ethos applies; these are the
JAX/state specifics that prevent real failures.

- **State lives in `GraphState`, not in objects.** Agents are pure factories; the
  evolving arrays live in `node_attrs` / `adj_matrices`. Don't add stateful classes.
- **Structural config is closed over; swept dials are 0-d arrays in
  `global_attrs`.** Config that fixes the program (sizes, horizons, which
  mechanisms exist) is closed over by transform factories. A dial to be swept is
  seeded as `jnp.array(cfg.x, dtype)` in `global_attrs` and read there (the `*_now`
  pattern, `ledger_society/state.py`), so `EnvSpec.run_grid` can vary it without
  recompiling. Python scalars never go in `global_attrs`: they are static pytree
  aux and every new value recompiles. Lifting can move batched numbers at float32
  rounding (XLA compiles a constant and an input differently): snapshot the batched
  paths before and after, and put the lift behind a flag if anything moved
  (`docs/run-grid-design.md`, "Lifting can move numbers").
- **No data-dependent Python control flow inside transforms.** Replace `if traced:`
  with `jnp.where` / `lax.cond` (e.g. `core.category.gated`) so it traces under
  `lax.scan`.
- **Pure tier vs eager tier.** Pure (`core.scan`) is the default and the only tier
  that `vmap`s over seeds. Use the eager tier (`core.time`) *only* for genuinely
  effectful agents (LLM/HTTP).
- **`EnvSpec.run` / `run_batch` compile once per spec.** Reuse the spec in a loop;
  a fresh `make_env`, `trace_fn` lambda or reducer dict per call is a fresh compile. For
  a dial grid, use `run_grid` (one compile per grid shape) instead of a `make_env`
  loop; compare its cells only with each other, within one call. The
  compiled program keeps whatever Python state `round_fn` read at first call, so
  closed-over config must be immutable. Numbers are bit-identical to the eager
  scan by construction (`EnvSpec._compiled`); keep `tests/test_spec_jit.py` exact.
- **Transforms declare `.reads` / `.writes`** via `@transform` so `compile_pipeline`
  can derive order. Same-family mechanisms keep disjoint writes.
- **The trajectory is the memory ceiling, not the state.** Raw per-agent traces cost
  O(T·N) — ~3000× the state at N≥500 — and no metric in this repo needs the joint
  (T, N) array. Reduce the *agent* axis in `trace_fn` (a (T,) scalar series is
  ~8 KB and worth keeping); fold per-agent or long-window quantities with a
  `metrics/reducers.py` `Reducer` via `EnvSpec.run_reduced` — O(1) carry, same
  number to float32 rounding. `examples/08_streaming_metrics.py` measures the gap.
- **One shared mathematical object = one piece of code.** A kernel asserted
  equivalent across environments is factored (see `environments/attachment.py`) or
  held equal by a bit-identity test — never by a comment.
- **Simplicity is a hard requirement.** Short, inspectable code; deletion over new
  abstraction; a catalog readable in one screen beats a clever registry framework.
- Assumptions cards (`ASSUMPTIONS.md` beside an environment) follow the prose rules
  in `docs/CLAUDE.md`.
