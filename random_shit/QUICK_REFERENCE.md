# Quick Reference: Agent System

## Basic Usage

```python
from core import GraphState, initialize_graph_state, get_observation, apply_action
from agents import RandomPolicy, TitForTatPolicy, LinearPolicy

# 1. Create graph state
state = initialize_graph_state(
    n_agents=5,
    n_resources=3,
    initial_resources=jnp.array(...)
)

# 2. Create policies
policy = RandomPolicy(output_shape=(n_agents, n_resources))

# 3. Simulation loop
for round in range(n_rounds):
    for agent_id in range(n_agents):
        # Get what agent observes
        obs = get_observation(state, agent_id)

        # Agent decides action
        action = policy(obs, key)

        # Apply action to state
        state = apply_action(state, agent_id, action)
```

## Files Created

**Core:**
- `core/agents.py` - Minimal agent interface (Policy, get_observation, apply_action)
- `core/initialization.py` - initialize_graph_state function

**Policies:**
- `agents/simple.py` - RandomPolicy, TitForTatPolicy
- `agents/learnable.py` - LinearPolicy

**Tests:**
- `quick_test_fishing.py` - Transfer semantics test
- `quick_test_fishing_v2.py` - Extraction semantics test
- `TEST_RESULTS.md` - Detailed findings
- `SUMMARY_FOR_JONAS.md` - Overview for you

## Architecture

```
GraphState (immutable state)
    ↓
get_observation(state, agent_id)
    ↓
observation matrix
    ↓
policy(obs, key)
    ↓
action matrix
    ↓
apply_action(state, agent_id, action)
    ↓
new GraphState
```

## Key Principles

1. **Pure functions**: No side effects, immutable updates
2. **Type explicit**: All functions typed with Protocol
3. **Composable**: Easy to swap policies, action semantics
4. **JAX-native**: Works with vmap, jit, grad

## Test Results

✅ Architecture works perfectly
✅ Observations extract correctly
✅ Actions update state functionally
✅ Policies compose cleanly

See `SUMMARY_FOR_JONAS.md` for full details.
