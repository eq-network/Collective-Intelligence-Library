# Getting Started: Build Your Own Simulation

## What You Actually Need (The 5 Core Pieces)

Ignore 90% of the democracy code. Here's the essential pattern:

### 1. **Define Your State** (5 minutes)

What does your simulation track? Define it in `GraphState`:

```python
from core.graph import GraphState

# Example: Social media influence simulation
def create_influence_state(num_users: int, seed: int):
    return GraphState(
        node_attrs={
            "influence_score": jnp.ones(num_users),      # Each user's influence
            "content_quality": jnp.random.uniform(0, 1, num_users),
            "is_verified": jnp.array([False] * num_users)
        },
        adj_matrices={
            "follows": jnp.zeros((num_users, num_users))  # Who follows whom
        },
        global_attrs={
            "round": 0,
            "total_engagement": 0.0,
            "seed": seed
        }
    )
```

**That's it for state.** GraphState is just a container. Put whatever you need in there.

---

### 2. **Write Transform Functions** (15 minutes per transform)

Transforms are just functions: `GraphState -> GraphState`

```python
from core.category import Transform

# Transform 1: Users follow influential people
def follow_influential_transform(state: GraphState) -> GraphState:
    influence = state.node_attrs["influence_score"]
    
    # Build follow matrix: follow top 20% most influential
    threshold = jnp.percentile(influence, 80)
    new_follows = influence[:, None] > threshold
    
    return state.update_adj_matrix("follows", new_follows)


# Transform 2: Influence spreads through network
def influence_spread_transform(state: GraphState) -> GraphState:
    follows = state.adj_matrices["follows"]
    influence = state.node_attrs["influence_score"]
    
    # Influence = own quality + 0.3 * average of people you follow
    neighbor_influence = jnp.matmul(follows, influence) / jnp.sum(follows, axis=1).clip(min=1)
    new_influence = 0.7 * state.node_attrs["content_quality"] + 0.3 * neighbor_influence
    
    return state.update_node_attrs("influence_score", new_influence)


# Transform 3: Verify top influencers
def verify_top_transform(state: GraphState) -> GraphState:
    influence = state.node_attrs["influence_score"]
    
    # Top 5% get verified
    threshold = jnp.percentile(influence, 95)
    verified = influence > threshold
    
    return state.update_node_attrs("is_verified", verified)
```

**Key insight**: These are just pure functions. No classes, no complicated setup.

---

### 3. **Compose Into Pipeline** (2 minutes)

Chain your transforms together:

```python
from core.category import sequential

# This is your simulation pipeline
influence_pipeline = sequential(
    follow_influential_transform,
    influence_spread_transform,
    verify_top_transform
)

# That's it. influence_pipeline is itself a Transform.
```

---

### 4. **Initialize & Run** (5 minutes)

```python
from execution.simulation import run_simulation
import jax.random as jr

# Create initial state
key = jr.PRNGKey(42)
initial_state = create_influence_state(num_users=100, seed=42)

# Run simulation
final_state, history = run_simulation(
    initial_state=initial_state,
    transform=influence_pipeline,
    num_rounds=20,
    key=key
)

# Check results
print(f"Final verified users: {jnp.sum(final_state.node_attrs['is_verified'])}")
```

---

### 5. **Extract Results** (optional, 5 minutes)

```python
# Get data from history
rounds = []
verified_counts = []
avg_influence = []

for state in history:
    rounds.append(state.global_attrs["round"])
    verified_counts.append(jnp.sum(state.node_attrs["is_verified"]))
    avg_influence.append(jnp.mean(state.node_attrs["influence_score"]))

# Plot it
import matplotlib.pyplot as plt
plt.plot(rounds, verified_counts)
plt.xlabel("Round")
plt.ylabel("Verified Users")
plt.show()
```

---

## That's The Whole Pattern

**File structure you need:**
```
your_simulation/
  __init__.py
  state.py          # Step 1: define create_your_state()
  transforms.py     # Step 2: write your transform functions
  pipeline.py       # Step 3: compose with sequential()
  run.py           # Step 4: run_simulation()
```

**Total code: ~100 lines.**

---

## What You Can Ignore From Democracy Code

### ❌ Don't Look At These (They're Domain-Specific):
- `/environments/noise_democracy/` - All democracy-specific
- `/environments/stable_democracy/` - All democracy-specific  
- `/transformations/democratic_transforms/` - Democracy voting logic
- Most of `mechanism_factory.py` - Complex democracy pipelines
- LLM integration code - Only needed if you use LLMs
- `optimality_analysis.py` - Democracy-specific metrics

### ✅ DO Use These:
- `/core/graph.py` - `GraphState` class
- `/core/category.py` - `sequential()`, `compose()`
- `/execution/simulation.py` - `run_simulation()`
- `/execution/call.py` - `execute()` (if you need custom execution)

---

## Common Patterns

### Pattern: Conditional Transforms
```python
def conditional_transform(condition_fn):
    def transform(state: GraphState) -> GraphState:
        if condition_fn(state):
            return some_transform(state)
        else:
            return state  # No-op
    return transform
```

### Pattern: Parameterized Transforms
```python
def create_spread_transform(spread_rate: float):
    def spread_transform(state: GraphState) -> GraphState:
        # Use spread_rate in your logic
        ...
    return spread_transform

# Use it:
pipeline = sequential(
    create_spread_transform(spread_rate=0.3),
    other_transform
)
```

### Pattern: Multi-Step Initialization
```python
def initialize_state(config):
    # Start with base state
    state = GraphState(
        node_attrs={"score": jnp.zeros(config.num_nodes)},
        adj_matrices={},
        global_attrs={}
    )
    
    # Apply initialization transforms
    state = setup_network_transform(state)
    state = assign_initial_scores_transform(state)
    
    return state
```

---

## Debugging Tips

### Print state between transforms:
```python
def debug_transform(state: GraphState) -> GraphState:
    print(f"Round {state.global_attrs['round']}")
    print(f"Node attrs: {state.node_attrs.keys()}")
    print(f"First 5 scores: {state.node_attrs['score'][:5]}")
    return state

pipeline = sequential(
    transform1,
    debug_transform,  # Insert debugging
    transform2
)
```

### Verify your state makes sense:
```python
def validate_state(state: GraphState):
    # Check shapes
    assert state.node_attrs["score"].shape == (100,)
    
    # Check ranges
    assert jnp.all(state.node_attrs["score"] >= 0)
    
    # Check conservation laws
    assert jnp.isclose(jnp.sum(state.node_attrs["resources"]), 1000.0)
```

---

## Example Domains You Could Build

**1. Epidemic Simulation**
- State: infection status per person, contact network
- Transforms: infection spread, recovery, vaccination

**2. Market Dynamics**
- State: trader positions, prices, order book
- Transforms: trading, price updates, market maker

**3. Content Moderation**
- State: post quality, user reports, moderator actions
- Transforms: content flagging, moderation decisions, appeals

**4. Traffic Flow**
- State: vehicle positions, speeds, traffic lights
- Transforms: vehicle movement, light changes, congestion

**5. Resource Allocation**
- State: agent resources, needs, trades
- Transforms: consumption, production, market exchange

---

## When You Need More

### Add Properties (Mathematical Guarantees)
```python
from core.property import ConservesSum

# Ensure total resources never change
resource_conservation = ConservesSum("resources")

# Attach to transform
from core.category import attach_properties
safe_transfer = attach_properties(
    transfer_transform, 
    {resource_conservation}
)
```

### Add Parallel Execution
```python
from execution.call import execute

final_state = execute(
    transform=your_pipeline,
    state=initial_state,
    spec={"strategy": "parallel", "num_workers": 4}
)
```

### Add Metrics Collection
```python
from execution.instrumentation.metrics import MetricsCollector

final_state, instrumentation = execute_with_instrumentation(
    transform=your_pipeline,
    state=initial_state,
    spec={"collect_metrics": True}
)

print(instrumentation["metrics"])
```

---

## The Mental Model

```
State → Transform → State → Transform → State → ...
```

That's it. Everything else is optimization or convenience.

**Your job:**
1. Define what's in State
2. Write functions that transform State
3. Chain them together
4. Run it

**The framework's job:**
- Provide immutable State container (GraphState)
- Provide composition operators (sequential, compose)
- Provide execution engine (run_simulation)

**You don't need**: LLMs, voting, elections, portfolios, predictions, delegations, or any of that. Those are just one example of transforms.

---

## Next Steps

1. **Copy the influence example above** into a new file
2. **Run it** - verify you get output
3. **Modify it** - change the transforms to do something different
4. **Add your domain logic** - replace influence with your concept

Once you can do that, you understand the pattern. Everything else is just more transforms.