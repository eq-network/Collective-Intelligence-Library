# Fishing Commons State Factory + Type Contracts

**Date:** 2026-03-30
**Files:** `experiments/fishing_commons/state.py` (new), `experiments/fishing_commons/mechanisms/__init__.py` (new)
**Task:** EXPR-01, MECH-04

## What Changed

### State Factory (`state.py`)

`create_initial_state(n_agents, adversarial_fraction, rng_key)` creates a GraphState for the fishing commons experiment:

- **node_types:** 0=cooperative, 1=adversarial (first n*α agents are adversarial)
- **node_attrs:** budget, harvest_weights, vote_weights, market_weights, signal, received_signals, allocations, last_harvest, reward
- **adj_matrices:** network (Erdos-Renyi, p=0.1, symmetric, no self-loops)
- **global_attrs (dynamic):** resource_level, harvest_target, penalty_lambda, clearing_price, rng_key, step
- **global_attrs (static):** K=5000, r=0.4, learning_rate=0.1, n_harvest_levels=6

### Type Contracts (`mechanisms/__init__.py`)

Documents the read/write interface for each mechanism type:

```python
MARKET_CONTRACT = {
    "reads": {"node_attrs": ["budget", "market_weights"], ...},
    "writes": {"node_attrs": ["allocations", "budget"], ...},
}
NETWORK_CONTRACT = { ... }
DEMOCRACY_CONTRACT = { ... }
```

## Design Decisions

- `harvest_levels` is a module-level constant (`DEFAULT_HARVEST_LEVELS`), NOT in `global_attrs`. It's configuration, not state — no reason to trace it.
- `rng_key` is in dynamic global_attrs. Mechanisms that need randomness must split and update it.
- Agent types are encoded as integers (0/1) in `node_types`, not as separate arrays.
- Network is symmetric (undirected) — `jnp.maximum(network, network.T)`.

## Reviewer Notes (unresolved)

- Missing type contracts for: resource dynamics, reward computation, agent learning
- Missing `cumulative_harvest` field for Gini coefficient metrics
- No protocol for how mechanisms coordinate RNG key splitting
- `adversarial_fraction` not stored in state (would be useful for analysis)
