# External API Guide

This guide helps external researchers (particularly those from RL/MARL backgrounds) understand how to use the Collective Intelligence Library.

---

## Key Difference: Messages → Messages

Traditional RL frameworks use an **Observation → Action** model:

```python
# Traditional RL (Gym-style)
observation = env.reset()
while not done:
    action = agent.act(observation)
    observation, reward, done, info = env.step(action)
```

The Collective Intelligence Library uses a **Messages → Messages** model:

```python
# Collective Intelligence Library
state = create_initial_state()
while not terminated:
    state = transform(state)
    # Agents are nodes in the graph
    # Their "observations" are incoming edges
    # Their "actions" are outgoing edges
```

---

## The Graph IS the Environment

There is no separate "environment" entity. Everything is represented in the `GraphState`:

| Traditional RL | Collective Intelligence Library |
|---------------|--------------------------------|
| `env.state` (hidden) | `state.node_attrs`, `state.adj_matrices` (explicit) |
| `env.agents` (external) | Agents are nodes in the graph |
| `action` (agent output) | Outgoing edges/messages |
| `observation` (env provides) | Incoming edges/messages |
| `reward` (env provides) | Computed from state (e.g., resource changes) |

---

## Basic Usage Pattern

### 1. Create Initial State

```python
import jax.numpy as jnp
from core.graph import GraphState

def create_my_state(num_agents: int) -> GraphState:
    return GraphState(
        node_types=jnp.zeros(num_agents, dtype=jnp.int32),
        node_attrs={
            "resources": jnp.ones(num_agents) * 100,
            "beliefs": jnp.zeros(num_agents),
        },
        adj_matrices={
            "communication": jnp.ones((num_agents, num_agents)),
        },
        global_attrs={
            "round": 0,
            "max_rounds": 100,
        }
    )
```

### 2. Define Agent Behavior

Agents are functions that produce actions based on observations:

```python
from typing import Dict, Any

Observation = Dict[str, Any]
Action = Dict[str, Any]

def simple_agent(observation: Observation, agent_id: int) -> Action:
    """A simple agent that shares resources with neighbors."""
    my_resources = observation["my_resources"]
    neighbor_resources = observation["neighbor_avg_resources"]

    if my_resources > neighbor_resources * 1.5:
        # Share if I have much more than average
        return {"share_amount": my_resources * 0.1}
    else:
        return {"share_amount": 0}
```

Or using the Agent base class:

```python
from engine.agents.base import Agent

class MyAgent(Agent):
    def act(self, observation: Observation) -> Action:
        # Your logic here
        return {"share_amount": 10}
```

### 3. Create a Transform

```python
from core.category import Transform
from core.graph import GraphState

def create_round_transform(agents: list) -> Transform:
    def transform(state: GraphState) -> GraphState:
        # 1. Create observations for each agent
        observations = []
        for i in range(state.num_nodes):
            obs = {
                "my_resources": float(state.node_attrs["resources"][i]),
                "neighbor_avg_resources": compute_neighbor_avg(state, i),
            }
            observations.append(obs)

        # 2. Get actions from agents
        actions = [agents[i](observations[i], i) for i in range(state.num_nodes)]

        # 3. Apply actions to state
        new_resources = state.node_attrs["resources"].copy()
        for i, action in enumerate(actions):
            share = action.get("share_amount", 0)
            new_resources = new_resources.at[i].add(-share)
            # Distribute to neighbors...

        # 4. Update round counter
        new_round = state.global_attrs["round"] + 1

        return state.update_node_attrs("resources", new_resources) \
                    .update_global_attr("round", new_round)

    return transform
```

### 4. Run the Simulation

```python
# Setup
state = create_my_state(num_agents=10)
agents = [simple_agent for _ in range(10)]
step = create_round_transform(agents)

# Run
history = [state]
for _ in range(100):
    state = step(state)
    history.append(state)

# Analyze
final_resources = state.node_attrs["resources"]
print(f"Final total resources: {jnp.sum(final_resources)}")
```

---

## Complete Example: Farmers Market

See `engine/environments/farmers_market/example.py`:

```python
from engine.environments.farmers_market.state import create_simple_farmers_market
from engine.environments.farmers_market.agent_transforms import create_agent_driven_round_transform
from engine.environments.farmers_market.agent_configs import (
    create_diversity_farmer,
    create_accumulator_farmer,
)

def run_simulation(num_farmers=10, num_rounds=50, seed=42):
    # Initial state
    state = create_simple_farmers_market(num_farmers, seed)

    # Heterogeneous agents
    agents = [
        create_diversity_farmer(),
        create_diversity_farmer(),
        create_accumulator_farmer(),
        create_accumulator_farmer(),
        # ... more agents
    ]

    # Transform
    step = create_agent_driven_round_transform(agents)

    # Run
    for _ in range(num_rounds):
        state = step(state)

    return state
```

---

## Mapping RL Concepts

### Observation Space

In RL, you define a fixed observation space. Here, observations are constructed from the graph:

```python
def get_observation(state: GraphState, agent_id: int) -> Observation:
    return {
        # Local node state (what the agent "owns")
        "my_resources": state.node_attrs["resources"][agent_id],
        "my_beliefs": state.node_attrs["beliefs"][agent_id],

        # Incoming edges (messages from neighbors)
        "incoming_messages": get_messages_to(state, agent_id),

        # Global information (if visible)
        "current_round": state.global_attrs["round"],
    }
```

### Action Space

Actions are dictionaries. The transform interprets them:

```python
# Example actions
action = {"bid": {"item": "apples", "price": 10, "quantity": 5}}
action = {"vote": {"choice": 2, "weight": 1.0}}
action = {"share": {"target": 3, "amount": 10}}
```

### Reward

Rewards are computed from state changes, not returned by the environment:

```python
def compute_reward(prev_state: GraphState, curr_state: GraphState, agent_id: int) -> float:
    prev_resources = prev_state.node_attrs["resources"][agent_id]
    curr_resources = curr_state.node_attrs["resources"][agent_id]
    return float(curr_resources - prev_resources)
```

### Done/Terminated

Check termination from global state:

```python
def is_done(state: GraphState) -> bool:
    return state.global_attrs["round"] >= state.global_attrs["max_rounds"]
```

---

## Gym-Style Wrapper (If Needed)

If you need a Gym-compatible interface:

```python
class GymWrapper:
    def __init__(self, initial_state_fn, transform_fn, agents):
        self.initial_state_fn = initial_state_fn
        self.transform = transform_fn
        self.agents = agents
        self.state = None

    def reset(self):
        self.state = self.initial_state_fn()
        return self._get_observations()

    def step(self, actions):
        # Apply actions via transform
        prev_state = self.state
        self.state = self.transform(self.state)

        observations = self._get_observations()
        rewards = self._compute_rewards(prev_state, self.state)
        done = self._is_done()
        info = {"round": self.state.global_attrs["round"]}

        return observations, rewards, done, info

    def _get_observations(self):
        return [get_observation(self.state, i) for i in range(self.state.num_nodes)]

    def _compute_rewards(self, prev, curr):
        return [compute_reward(prev, curr, i) for i in range(curr.num_nodes)]

    def _is_done(self):
        return is_done(self.state)
```

---

## Key Differences Summary

| Aspect | Traditional RL | Collective Intelligence Library |
|--------|---------------|--------------------------------|
| State location | Hidden in env | Explicit in GraphState |
| Agent location | External to env | Nodes in the graph |
| Communication | None (independent) | Edges carry messages |
| Observation | Env provides | Constructed from edges |
| Action | Discrete/Continuous | Typed messages |
| Reward | Env provides | Computed from state |
| Step | `env.step(action)` | `state = transform(state)` |

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System overview
- [HETEROGENEOUS_NODES.md](./HETEROGENEOUS_NODES.md) - Node types (Agent, Market, Democracy)
- `engine/environments/farmers_market/` - Complete working example
