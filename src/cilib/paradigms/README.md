# Paradigms — how to import a simulation paradigm

A **paradigm** is a self-contained simulation model (a set of agents + a world +
dynamics) packaged so it plugs into the core framework without touching the core.
The point: a new paper/study becomes a *folder here*, reusing the engine
(`GraphState`, `Transform`, `core.scan.run_scan`, vectorized message passing)
instead of a bespoke repo. The first paradigm, `active_inference/`, is the worked
example — read it alongside this contract.

## The contract (6 parts)

A paradigm `engine/paradigms/<name>/` provides:

1. **`schema.py`** — the state layout. Which `node_attrs` each agent carries
   (leading axis = N agents), which `adj_matrices` (e.g. `trust`, with a positive
   diagonal = memory/self-loop), and an `*Config` frozen dataclass of the *static*
   scenario constants. Constants are **closed over by the transforms, never stored
   in `GraphState.global_attrs`** (which is static pytree aux — putting per-step or
   swept data there forces recompiles). `GraphState` carries only evolving arrays.

2. **`primitives/`** — pure math kernels (pure functions on bare arrays, no
   scenario knowledge). Vendor here anything worth lifting exactly. Batching over
   agents/candidates is the caller's job (`jax.vmap` in `transforms.py`).

3. **`transforms.py`** — the round steps as `make_*(cfg)` factories returning pure
   functions. Each step is vectorized (vmap/einsum), so it runs inside
   `core.scan.run_scan` and `vmap`s over seeds. Compose with `core.category`'s
   `sequential` and `gated` (scan-safe conditional — an "ablation flag" is just the
   presence/absence of a step). A step that needs the tick/key has signature
   `(state, t, key) -> state`; a pure transform is `state -> state`.

4. **`agents.py`** — `PureAgent` factories. A pure agent does **not** hold state
   (that lives in `GraphState`); it exposes `round_fn() -> (state, t, key) -> state`,
   the pure round it implements. (Contrast the effectful `engine.agents.llm_agent`,
   which brackets an HTTP call and must run on the eager `core.time` tier.)

5. **`environments/`** — `create_initial_state(**params)` scenario factories
   (implements `core.environment.Environment`). Build the initial `GraphState` and
   the matching `*Config`. Swept dials are plain function params (→ `vmap` axes on
   the fast path, or `execution.config.ExperimentConfig.parameter_sweeps` for the
   eager/heterogeneous path).

6. **`tests/`** — behavioral acceptance signatures. Assert the *mechanism*
   (qualitative phenomena), not bit-exact numbers, run through `run_scan`.

## Running a paradigm (the fast path)

```python
from core.scan import run_scan, run_scan_batch
cfg, state, T = <env_preset>(...)
round_fn = <Agent>(cfg).round_fn()

# one run
final, trace = run_scan(round_fn, state, T, key, trace_fn=<readout>)

# a whole sweep: vmap over seeds -> ONE compiled program
finals, traces = run_scan_batch(round_fn, lambda k: state, T, jax.random.split(key, 256))
```

## Two tiers (where your paradigm lives)

- **Pure tier** (`core.scan`): every agent + the world is side-effect-free. The
  round is pure JAX → compiled `lax.scan`, `vmap` over seeds/dials as one program.
  This is the default and where research paradigms (incl. `active_inference`) live.
- **Eager tier** (`core.time`: `World`/`tick_world`): for genuinely effectful agents
  (live LLM/sensor/world calls). Slower, not `vmap`-able over the effect; reach for
  it only when a node must call the outside world mid-loop.

The boundary is a property of the agent (pure vs effectful), not of the engine.
