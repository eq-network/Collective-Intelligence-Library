# Collective Intelligence Library: Process-Centric Multi-Agent Simulation

A JAX-native functional framework for building multi-agent simulations using graph transformations and category theory. Built for mechanism-design experiments in computational social science and multi-agent AI.

> ### Status: research code, actively evolving
>
> This is a research codebase. It is **not** a polished library — we are figuring things out as we go, and specifics will change.
>
> **What's stable (the load-bearing ideas):**
> - Functional programming foundation: state is immutable, transforms are pure functions.
> - The core trio: a typed state, transforms over that state, and sequential / compositional assembly of those transforms into mechanisms.
> - Online / `lax.scan`-style execution: simulations are a scan of transforms across ticks, not an imperative update loop.
>
> **What will change (details in flux):**
> - Concrete data types (`GraphState` field names, pytree layout, what's dynamic vs static).
> - The compiler / pipeline machinery (right now a read/write-set DAG, but this is an implementation choice, not a commitment).
> - Naming, module boundaries, APIs of individual transforms.
>
> Documentation lags the code. Some things that live in Jonas' head aren't written down anywhere yet. **If you're interested in the ideas, the direction, or collaborating — please reach out.** The codebase will make more sense after a short conversation than after a long read.
>
> Contact: [Jonas Hallgren](https://github.com/spiralling) · Uppsala University · `jonas@eq-network.org`

> **State of the library (synced 2026-04-16)**
> - **Stable enough to use:** `core/` (GraphState, typed transforms, `sequential/parallel/conditional`, pipeline compiler, schedule primitive, `lax.scan` environment); `metrics/` families; `vmap`-over-seeds execution.
> - **Flagship experiment:** `experiments/basin_stability/` — PDD / PRD / PLD under adversarial pressure on a proposal-selection resource game. See [`AGENT_ARCHITECTURE.md`](experiments/basin_stability/AGENT_ARCHITECTURE.md).
> - **Reference examples (may lag the core API):** `experiments/fishing_commons/`, `experiments/governed_harvest/`.
> - **Next milestones:** full 4,200-condition vmap sweep on GPU; thesis figure set.

## What Is This?

The Collective Intelligence Library treats simulations as **pure transformations on graph states** rather than object-oriented state mutations. This makes complex multi-agent systems easier to reason about, compose, and verify.

**Core Principle**: `GraphState → Transform → GraphState → Transform → ...`

That's it. Everything else is optimization or convenience.

**Why you might care (by field):**
- *Computational social science:* swap aggregation rules, delegation topologies, or election schedules without rewriting the game. Mechanisms compose.
- *Multi-agent AI:* agents are pure functions over a shared `GraphState`; runs are batched via `jax.vmap`, making 1000-seed adversarial sweeps feasible on a single GPU.
- *Mechanism design:* every pipeline carries declared read/write sets, so institutional dependencies are a typed DAG, not a tangle of update hooks.

## Getting Started

If you're new to the framework, start with the [Start Here Guide](Start_Here.md) for a hands-on walkthrough, and the [Manifesto](Manifesto.md) for the conceptual foundations. For the architectural reasoning behind the current primitives — with TikZ diagrams — see [`changelogs/2026-03-30_core-primitives.pdf`](changelogs/2026-03-30_core-primitives.pdf).

The flagship worked example lives in `experiments/basin_stability/`. The core framework is in `core/` and `metrics/`, and you can use it to build simulations for any domain: epidemics, markets, opinion dynamics, or anything else involving agents and networks.

## Installation

```bash
git clone https://github.com/eq-network/Col-Int-Lib.git
cd Col-Int-Lib
pip install -r requirements.txt
# or, for editable install with JAX CUDA 12:
pip install -e ".[cuda]"
```

**Requirements**: Python 3.10+, JAX 0.4.20+, NumPy, matplotlib. GPU optional (via `jax[cuda12]`).

## Quick Example

```python
import jax.numpy as jnp
from core.graph import GraphState
from core.category import sequential

# Define your state (JAX arrays, not numpy — JIT/vmap compatibility)
state = GraphState(
    node_types=jnp.zeros(100, dtype=jnp.int32),
    node_attrs={"score": jnp.ones(100)},
    adj_matrices={"network": jnp.zeros((100, 100))},
    global_attrs={"round": jnp.array(0)},   # dynamic (traced)
)

# Transforms are pure GraphState -> GraphState
def update_scores(state):
    return state.update_node_attrs("score", state.node_attrs["score"] * 1.1)

def update_round(state):
    return state.update_global_attr("round", state.global_attrs["round"] + 1)

# Compose and run
pipeline = sequential(update_scores, update_round)
final_state = pipeline(state)
```

For a full experiment (with metrics, vmap-over-seeds, and adversarial sweeps), see [`experiments/basin_stability/run_experiment.py`](experiments/basin_stability/run_experiment.py). The [Start Here Guide](Start_Here.md) walks through the same pattern pedagogically.

### Run the flagship experiment

```bash
# Quick smoke test (10 seeds per condition, sequential)
python -m experiments.basin_stability.run_experiment --quick

# Full sweep (100 seeds × 3 mechanisms × 2 tracking modes × 7 adversarial fractions, vmap-batched)
python -m experiments.basin_stability.run_experiment --n_seeds 100 --vmap --plot
```

## Repository Structure

```
collective-intelligence-library/
├── core/                 # The framework itself
│   ├── graph.py             # GraphState (JAX pytree, dynamic/static partition)
│   ├── category.py          # @transform, sequential, parallel, conditional, identity
│   ├── pipeline.py          # compile_pipeline: derive execution order from read/write DAG
│   ├── schedule.py          # ScheduleEntry + make_scheduled_step (cadence as variable)
│   └── environment.py       # Environment: lax.scan harness
├── metrics/              # Composable metric families (economic / governance / graph)
├── experiments/          # Worked examples and research experiments
│   ├── basin_stability/     # Flagship: PDD/PRD/PLD vs adversarial pressure
│   ├── fishing_commons/     # State factory + type contracts reference
│   └── governed_harvest/    # Earlier harvest-extraction prototype
├── plans/                # Design docs: ARCHITECTURE, NEXT_STEPS, experiment plans
├── changelogs/           # Per-change notes (.md) + typeset TikZ figures (.tex/.pdf)
└── Manifesto.md          # The "why" — long-form philosophical argument
```

For your own simulation you'll primarily touch `core/` (state + composition), `metrics/` (instrumentation), and create a new folder under `experiments/`. The directory layout to copy is `basin_stability/`: `state.py` (`create_initial_state`), `policies.py` (pure vmappable functions), `transforms.py` (decorated transforms + `make_step_transform`), `environment.py` (subclass `Environment` + `run_batched`), `run_experiment.py` (sweep + CSV export).

## Core Concepts (5 Minutes)

### 1. GraphState

GraphState is an immutable container for your simulation state. It holds per-agent data in `node_attrs`, network structure in `adj_matrices`, and system-level data in `global_attrs`:

```python
state = GraphState(
    node_types=np.array([...]),                   # Node type labels
    node_attrs={"resources": np.array([...])},    # Per-agent data
    adj_matrices={"connections": np.array([...])}, # Network structure
    global_attrs={"round": 0}                       # System-level data
)
```

### 2. Transforms

Transforms are pure functions that map from one GraphState to another. They read from the input state, compute new values, and return a new state without ever mutating the original:

```python
def my_transform(state: GraphState) -> GraphState:
    # Read from state, compute new values
    # Return new state (never mutate!)
    return state.update_node_attrs("score", new_scores)
```

### 3. Composition

Complex behaviors emerge from chaining simple transforms together. The `sequential` function composes multiple transforms into a pipeline:

```python
from core.category import sequential

pipeline = sequential(
    information_spread,
    belief_update,
    network_rewiring
)

final_state = pipeline(initial_state)
```

That's the entire pattern. Everything else builds on this foundation.

## The Basin Stability Example

The current flagship application lives in `experiments/basin_stability/`. It tests three democratic mechanisms — PDD (direct), PRD (representative), PLD (liquid) — under adversarial pressure on a common-pool resource game, and compares *predictive* vs *non-predictive* trust dynamics. [`AGENT_ARCHITECTURE.md`](experiments/basin_stability/AGENT_ARCHITECTURE.md) documents exactly what each agent does and why.

This is an example, not part of the framework. If you're building a simulation in a different domain, use it as a template for what a full experiment looks like (state factory, pure policy functions, decorated transforms, a composed step pipeline, a vmap sweep, CSV export).

## Philosophy

This framework embraces functional purity, where transformations have no side effects. States are immutable — never modified, only transformed into new states. Complex behaviors emerge through composition of simple parts. Mathematical properties are encoded in the type system, and there's a clean separation between what transformations do mathematically and how they execute computationally.

Read the [Manifesto](Manifesto.md) for the full philosophical and technical argument.

## Documentation

| Doc | Purpose | Last synced |
|---|---|---|
| [Start_Here.md](Start_Here.md) | Hands-on walkthrough for a first simulation | *pending refresh* |
| [Manifesto.md](Manifesto.md) | The "why": process-centric thinking + category theory framing | 2026-02 |
| [plans/](plans/) | Design docs + experiment plans (ARCHITECTURE, NEXT_STEPS, basin_stability) | 2026-04 |
| [changelogs/](changelogs/) | Per-change notes (Markdown) + typeset figures (LaTeX/PDF) | 2026-03 |
| [experiments/basin_stability/AGENT_ARCHITECTURE.md](experiments/basin_stability/AGENT_ARCHITECTURE.md) | Agent-level spec for the flagship experiment | 2026-04-16 |
| [Excalidraw diagrams](https://excalidraw.com/#room=f4116b0ba2d8d5095d85,zSDwGDuqMZI4uxu4CTQuHg) | Live visual architecture overview | continuous |

### Documentation & update cadence

To keep this readable from outside without pretending it's a finished product, we follow a light rhythm:

- **Per-change notes go in `changelogs/` as Markdown** on the day the change lands. These are short, technical, and dated (`YYYY-MM-DD_what-changed.md`).
- **Monthly roll-up as LaTeX/PDF.** At the end of each month, the Markdown changelogs covering architectural primitives get compiled into a single typeset document with TikZ diagrams — see [`changelogs/2026-03-30_core-primitives.pdf`](changelogs/2026-03-30_core-primitives.pdf) as the template. Read this if you want the *why* behind the current primitives.
- **README "State of the library" box refreshed at each monthly roll-up.** So if the date at the top of this README is more than ~6 weeks old, assume the box is out of sync with the code and skim recent `changelogs/` instead.
- **Plans in `plans/` are living documents.** They describe intent and may run ahead of what's implemented — treat them as direction-setting, not as specification.
- **Versioning:** `0.x` pre-thesis. `0.1` = basin_stability sweep end-to-end. `1.0` will mean the thesis figures are reproducible from a tagged commit. We are at the `0.1` milestone.

If something in the docs contradicts the code, **the code is the source of truth** and the docs are drifting. Opening an issue or pinging Jonas is the fastest fix.

## Contributing

This is research code that's changing rapidly. If you're interested in building your own domain simulations, improving the core framework, or adding new transformation patterns, please reach out or submit a PR.

## License

MIT

---

**Remember**: `basin_stability` is one worked example. The framework is for building **any** graph-based multi-agent simulation where you want typed, composable mechanisms. Start with [Start_Here.md](Start_Here.md) and build something new.