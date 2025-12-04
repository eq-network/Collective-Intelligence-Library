# Architecture Overview

The Collective Intelligence Library is a programming language for coordination mechanisms built on graph transformations.

## Core Principle: The Graph IS the Environment

Unlike traditional RL frameworks where an "environment" contains agents, here **everything is the graph**:

```
Traditional RL:                    Collective Intelligence Library:
┌─────────────────┐                ┌─────────────────────────────────┐
│   Environment   │                │                                 │
│  ┌───────────┐  │                │         GraphState              │
│  │  Agent 1  │  │                │   ┌───┐     ┌───┐     ┌───┐    │
│  │  Agent 2  │  │                │   │ A │────→│ M │────→│ D │    │
│  │    ...    │  │                │   └───┘     └───┘     └───┘    │
│  └───────────┘  │                │     ↑         ↑         ↑      │
│   (separate)    │                │   Node      Node      Node     │
└─────────────────┘                │  (Agent)  (Market) (Democracy) │
                                   │                                 │
                                   │   Everything IS the graph       │
                                   │   Edges = information flow      │
                                   └─────────────────────────────────┘
```

- **Nodes** represent entities (Agents, Markets, Democracies)
- **Edges** represent information flow (all communication)
- **Transforms** operate on the entire graph state

---

## The 5-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4: EXECUTION                                                  │
│  ═══════════════════════════════════════════════════════════════    │
│  Environment wrapper, external APIs, experiment orchestration        │
│  Files: core/environment.py, core/simulation.py                     │
│                                                                      │
│  Purpose: Interface for external systems (RL researchers, etc.)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Layer 3: BEHAVIORS                                                  │
│  ═══════════════════════════════════════════════════════════════    │
│  NodeBehavior protocols for heterogeneous node types                 │
│  Files: engine/agents/, core/agents.py                              │
│                                                                      │
│  Purpose: Define how different node types process messages           │
│  - Agent nodes: Produce actions based on observations                │
│  - Market nodes: Aggregate bids into prices                         │
│  - Democracy nodes: Aggregate votes into decisions                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Layer 2: COMPOSITION                                                │
│  ═══════════════════════════════════════════════════════════════    │
│  sequential(), compose(), conditional()                              │
│  File: core/category.py                                             │
│                                                                      │
│  Purpose: Combine transforms into complex pipelines                  │
│                                                                      │
│    pipeline = sequential(                                            │
│        message_passing,                                              │
│        market_clearing,                                              │
│        trust_update                                                  │
│    )                                                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Layer 1: TRANSFORMS                                                 │
│  ═══════════════════════════════════════════════════════════════    │
│  Pure functions: GraphState → GraphState                             │
│  Files: engine/transformations/                                     │
│                                                                      │
│  Purpose: Individual operations on graph state                       │
│  - Message passing (bottom-up)                                       │
│  - Resource allocation (top-down)                                    │
│  - State updates                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Layer 0: DATA                                                       │
│  ═══════════════════════════════════════════════════════════════    │
│  GraphState (immutable JAX arrays)                                   │
│  File: core/graph.py                                                │
│                                                                      │
│  Structure:                                                          │
│    node_types:   [N]       - Type label for each node               │
│    node_attrs:   {name: [N]}     - Per-node attributes (diagonal)   │
│    adj_matrices: {name: [N×N]}   - Connectivity (off-diagonal)      │
│    global_attrs: {name: value}   - System-level metadata            │
│                                                                      │
│  Key insight: Immutability enables parallelization                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How the Layers Connect

### Data → Transform → Data

The fundamental computation is:

```python
new_state = transform(current_state)
```

All transforms are pure functions. They read from `current_state` and return a completely new `new_state`. The original is never modified.

### Composition Builds Complexity

Simple transforms compose into complex behaviors:

```python
# Each of these is GraphState → GraphState
information_sharing = create_message_passing_transform(...)
market_clearing = create_market_transform(...)
trust_update = create_trust_transform(...)

# Composition creates a pipeline
full_round = sequential(
    information_sharing,
    market_clearing,
    trust_update
)

# Still just GraphState → GraphState
final_state = full_round(initial_state)
```

### Behaviors Define Node Logic

Different node types process messages differently:

| Node Type | Receives | Produces |
|-----------|----------|----------|
| Agent | Messages from neighbors | Actions/Messages |
| Market | Bids from agents | Prices, allocations |
| Democracy | Votes from agents | Decisions, policies |

All operate within the same graph structure - just with different processing logic.

### Execution Wraps Everything

The Environment class (Layer 4) is just an execution wrapper that:
1. Initializes the graph
2. Runs transform pipelines in a loop
3. Records history for analysis

It doesn't "contain" the simulation - the GraphState does.

---

## File Reference

| Layer | Primary Files | Purpose |
|-------|--------------|---------|
| 4 - Execution | `core/environment.py`, `core/simulation.py` | External API, orchestration |
| 3 - Behaviors | `core/agents.py`, `engine/agents/` | Node-type-specific logic |
| 2 - Composition | `core/category.py` | `sequential()`, `compose()` |
| 1 - Transforms | `engine/transformations/` | Message passing, markets, etc. |
| 0 - Data | `core/graph.py` | `GraphState` definition |

---

## Key Design Decisions

1. **Immutability First**: All state updates return new GraphState. This enables JAX parallelization.

2. **Graph as Substrate**: Everything is represented in the graph. No hidden state.

3. **Edges are Information**: All communication happens via edges. Types of edges determine message protocols.

4. **Transforms are Composable**: Any transform can be combined with any other via `sequential()` or `compose()`.

5. **Heterogeneous Nodes**: Agents, Markets, and Democracies are all nodes with different behaviors - not separate systems.

---

## See Also

- [COMPUTATIONAL_FOUNDATION.md](./COMPUTATIONAL_FOUNDATION.md) - Why immutability enables parallelization
- [DESIGN_PATTERNS.md](./DESIGN_PATTERNS.md) - SOLID principles in the codebase
- [HETEROGENEOUS_NODES.md](./HETEROGENEOUS_NODES.md) - Agent/Market/Democracy node types
- [EXTERNAL_API.md](./EXTERNAL_API.md) - Guide for external researchers
