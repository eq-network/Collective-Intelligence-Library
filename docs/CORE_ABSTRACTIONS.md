# Core Abstractions

This document defines the minimal, correct abstractions that form the foundation of Mycorrhiza.

## Design Philosophy

`core/` defines the **type system** for the collective intelligence language:

- **Minimal**: Only essential abstractions
- **Pure**: No mutable state
- **Composable**: Everything works together cleanly
- **Pedagogical**: Reference implementations others can learn from

Think of `core/` like the type signatures in a functional programming language - they define what's possible, not how to implement it.

---

## The Four Core Abstractions

### 1. GraphState (Data)

**File**: `core/graph.py`

**What**: Immutable graph representation

```python
@dataclass(frozen=True)
class GraphState:
    node_types: jnp.ndarray              # [N] Node type labels
    node_attrs: Dict[str, jnp.ndarray]   # Per-node attributes
    adj_matrices: Dict[str, jnp.ndarray] # Connectivity
    global_attrs: Dict[str, Any]         # System metadata
```

**Key Properties**:
- Immutable (`frozen=True`)
- Update methods return new instances
- Matrix-based for JAX parallelization

---

### 2. Transform (Computation)

**File**: `core/category.py`

**What**: Pure function on graphs

```python
Transform = Callable[[GraphState], GraphState]
```

**Key Properties**:
- Pure: no side effects
- Composable: `compose(f, g)`, `sequential(*transforms)`
- Can attach properties: `attach_properties(transform, {ConservesSum("resources")})`

---

### 3. Agent (Behavior)

**File**: `core/agents.py`

**What**: Decision-making protocol

```python
class Agent(Protocol):
    def act(self, observation: Observation) -> Action:
        """Observation → Action"""
```

**Key Properties**:
- Minimal protocol (just `act` method)
- No agent_id in protocol (managed by graph structure)
- Supports both functions and classes
- Can be stateful or stateless

**Examples**:

```python
# Function-based (stateless)
def simple_agent(observation: Observation) -> Action:
    return {"share": observation["resources"] * 0.1}

# Class-based (can be stateful)
class LearningAgent:
    def __init__(self):
        self.memory = []

    def act(self, observation: Observation) -> Action:
        self.memory.append(observation)
        return self.make_decision()
```

---

### 4. Environment (Scenario Factory)

**File**: `core/environment.py`

**What**: Creates initial GraphState for scenarios

```python
class Environment(Protocol):
    def create_initial_state(self, **params) -> GraphState:
        """Factory for scenario initialization"""
```

**Key Properties**:
- NOT a stateful container (unlike OpenAI Gym)
- Just creates initial state
- Like scenarios: FarmersMarket, TragedyOfCommons, etc.

**Example**:

```python
class FarmersMarket:
    def create_initial_state(self, num_farmers: int, seed: int) -> GraphState:
        return GraphState(
            node_types=jnp.zeros(num_farmers),
            node_attrs={"resources_apples": jnp.ones(num_farmers) * 10},
            adj_matrices={"trade": jnp.eye(num_farmers)},
            global_attrs={"round": 0}
        )

# Usage
env = FarmersMarket()
initial_state = env.create_initial_state(num_farmers=10, seed=42)

# Environment is done - now just use transforms
transform = create_market_transform()
state = initial_state
for _ in range(100):
    state = transform(state)
```

---

## Property System

**File**: `core/property.py`

**What**: Verify mathematical invariants

```python
class Property:
    def check(self, state: GraphState) -> bool:
        """Verify property holds on state"""
```

### ConservesSum (Fixed)

The most important property - verifies resource conservation:

```python
# Create property
conservation = ConservesSum("resources")

# Bind to initial state
bound = conservation.bind(initial_state)

# Verify after transform
final_state = transform(initial_state)
assert bound.check(final_state)  # Verifies sum unchanged
```

**Why binding?** The property needs a reference to compare against. Without binding, it can't know if the sum changed.

---

## How Abstractions Compose

### Example: Resource Sharing Game

```python
# 1. Environment creates initial state
env = SimpleResourceGame()
state = env.create_initial_state(num_agents=5, resources=100)

# 2. Agents implement behavior protocol
agents = [GenerousAgent(), GreedyAgent(), RandomAgent()]

# 3. Transform combines agent behaviors
def create_round_transform(agents: list[Agent]) -> Transform:
    def transform(state: GraphState) -> GraphState:
        # Extract observations
        observations = [extract_obs(state, i) for i in range(state.num_nodes)]

        # Get actions
        actions = [agents[i].act(observations[i]) for i in range(len(agents))]

        # Apply actions (pure update)
        return apply_actions(state, actions)

    return transform

# 4. Run simulation (pure functional loop)
step = create_round_transform(agents)
for _ in range(100):
    state = step(state)

# 5. Verify properties
conservation = ConservesSum("resources").bind(initial_state)
assert conservation.check(state)
```

---

## Architecture Decisions

### Why Environment is a Protocol, Not a Class?

**Old approach** (stateful container):
```python
class Environment:
    def __init__(self, initial_state):
        self.state = initial_state  # X Mutable!
        self.round_num = 0
        self.history = []

    def step(self):
        self.state = transform(self.state)  # X Mutation!
```

**New approach** (scenario factory):
```python
class FarmersMarket:
    def create_initial_state(self, **params) -> GraphState:
        return GraphState(...)  # ✓ Pure function

# No state stored - environment just creates initial GraphState
```

**Benefits**:
1. **No mutable state** - aligns with functional principles
2. **Simpler** - just a factory, not a runtime container
3. **Composable** - states flow through transforms
4. **Parallelizable** - no shared mutable state

### Why Agent Has No agent_id?

Agent ID is a property of the **graph structure**, not the agent:

```python
# Bad: ID in agent
class Agent:
    def __init__(self, agent_id: int):  # X Couples agent to position
        self.agent_id = agent_id

# Good: ID from graph
agents = [Agent1(), Agent2(), Agent3()]  # ✓ Just behaviors
# Position in list = node index in graph
```

**Benefits**:
1. Agents are **portable** - same agent can be at different positions
2. Agents are **composable** - just functions or callable objects
3. Graph manages structure, agents manage behavior

---

## Where Things Live

| Abstraction | Core (type) | Engine (implementations) |
|-------------|-------------|-------------------------|
| **GraphState** | `core/graph.py` | `engine/environments/*/state.py` |
| **Transform** | `core/category.py` | `engine/transformations/*` |
| **Agent** | `core/agents.py` | `engine/agents/*` |
| **Environment** | `core/environment.py` | `engine/environments/*` |
| **Property** | `core/property.py` | Used in validation |

---

## Example: See It In Action

Run the clean architecture demo:

```bash
python examples/clean_architecture_demo.py
```

This shows:
1. Environment as scenario factory
2. Agent as minimal protocol (3 different agent types)
3. Transform composition
4. Property verification (resources conserved)

---

## Summary

The core abstractions are:

1. **GraphState**: Immutable data
2. **Transform**: Pure functions `GraphState → GraphState`
3. **Agent**: Behavior protocol `Observation → Action`
4. **Environment**: Scenario factory `params → GraphState`

Everything else is built from these primitives through composition.

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - 5-layer system overview
- [COMPUTATIONAL_FOUNDATION.md](./COMPUTATIONAL_FOUNDATION.md) - Why immutability enables parallelization
- [DESIGN_PATTERNS.md](./DESIGN_PATTERNS.md) - SOLID principles analysis
- `examples/clean_architecture_demo.py` - Working example
