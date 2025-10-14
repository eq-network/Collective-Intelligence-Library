# Collective Intelligence Library: Process-Centric Multi-Agent Simulation

A functional framework for building multi-agent simulations using graph transformations and category theory.

## What Is This?

The Collective Intelligence Library treats simulations as **pure transformations on graph states** rather than object-oriented state mutations. This makes complex multi-agent systems easier to reason about, compose, and verify.

**Core Principle**: `GraphState → Transform → GraphState → Transform → ...`

That's it. Everything else is optimization or convenience.

## Getting Started

If you're new to the framework, start with the [Start Here Guide](Start_Here.md) which walks you through building your first simulation in about 30 minutes. For a deeper understanding of the conceptual foundations and design philosophy, read the [Manifesto](Manifesto.md).

The democracy code in `environments/democracy/` is just one example application. The core framework lives in `core/`, `engine/`, and `execution/`, and you can use it to build simulations for any domain: epidemics, markets, traffic flow, or anything else involving agents and networks.

## Installation

```bash
git clone https://github.com/eq-network/Col-Int-Lib.git
cd Col-Int-Lib
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, NumPy, pandas

## Quick Example

```python
from core.graph import GraphState
from core.category import sequential
import numpy as np

# Define your state
state = GraphState(
    node_types=np.zeros(100),
    node_attrs={"score": np.ones(100)},
    adj_matrices={"network": np.zeros((100, 100))},
    global_attrs={"round": 0}
)

# Write transform functions
def update_scores(state):
    new_scores = state.node_attrs["score"] * 1.1
    return state.update_node_attrs("score", new_scores)

def update_round(state):
    return state.update_global_attr("round", state.global_attrs["round"] + 1)

# Compose and run
pipeline = sequential(update_scores, update_round)
final_state = pipeline(state)
```

See [Start_Here.md](Start_Here.md) for a complete tutorial.

## Repository Structure

```
collective-intelligence-library/
├── core/              # Core framework (GraphState, Transform, Property)
├── engine/            # Domain-agnostic transformation building blocks
│   ├── transformations/
│   │   ├── bottom_up/    # Agent-level transformations
│   │   └── top_down/     # System-level mechanisms
│   └── agents/           # Agent implementation patterns
├── execution/         # Execution strategies and analysis tools
├── environments/      # Example: Democracy simulations (ignore for new domains)
└── services/          # Optional: LLM integration, etc.
```

For your own simulation, you'll primarily work with `core/` (the graph transformation system), `engine/transformations/` (reusable transformation patterns), and `execution/` (running simulations and analyzing results). You can safely ignore `environments/democracy/` unless you're specifically interested in that example, and most of `services/` unless you need LLM integration.

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

## The Democracy Example

The `environments/democracy/` code demonstrates these concepts through voting simulations, but it's not part of the core framework. It's one example of how to use the library. If you're building your own simulation, you can safely ignore the democracy code and focus on the core framework.

## Philosophy

This framework embraces functional purity, where transformations have no side effects. States are immutable—never modified, only transformed into new states. Complex behaviors emerge through composition of simple parts. Mathematical properties are encoded in the type system, and there's a clean separation between what transformations do mathematically and how they execute computationally.

Read the [Manifesto](Manifesto.md) for the full philosophical and technical argument.

## Documentation

The [Start Here](Start_Here.md) guide walks you through building your first simulation. The [Manifesto](Manifesto.md) explains the conceptual foundations and design principles. You can also view the [Excalidraw Diagrams](https://excalidraw.com/#room=f4116b0ba2d8d5095d85,zSDwGDuqMZI4uxu4CTQuHg) for visual architecture overview.

## Contributing

This is research code that's changing rapidly. If you're interested in building your own domain simulations, improving the core framework, or adding new transformation patterns, please reach out or submit a PR.

## License

MIT

---

**Remember**: The democracy code is just an example. The framework is for building **any** graph-based multi-agent simulation. Start with [Start_Here.md](Start_Here.md) and build something new.