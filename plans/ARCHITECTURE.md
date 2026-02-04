# Mycorrhiza Architecture

## Design Philosophy

**Core Principle:** Everything is a pure function from `GraphState → GraphState`.

### Why Functional?

1. **Reasoning** - No hidden state, clear data flow
2. **Parallelization** - No mutation = no race conditions
3. **Testing** - Deterministic, reproducible
4. **Composition** - Functions chain naturally

### Why JAX?

1. **vmap** - Automatic vectorization over batch dimensions
2. **jit** - Just-in-time compilation for speed
3. **grad** - Automatic differentiation for learning
4. **GPU** - Same code runs on GPU with zero changes

## Core Data Structures

### GraphState

```python
@dataclass
class GraphState:
    node_types: jnp.ndarray              # (n_agents,) - Agent type labels
    node_attrs: Dict[str, jnp.ndarray]   # e.g., {"resources": (n_agents, n_res)}
    edge_attrs: Dict[str, jnp.ndarray]   # e.g., {"messages": (n, n, n_res)}
    adj_matrices: Dict[str, jnp.ndarray] # e.g., {"connections": (n, n)}
    global_attrs: Dict[str, Any]         # e.g., {"round": 0}
```

**Design choices:**
- **Immutable** - Updates return new state
- **Dict-based** - Flexible attribute storage
- **JAX arrays** - Works with vmap/jit/grad
- **Multiple adjacencies** - Different relationship types

**Update methods:**
```python
# Node attributes
new_state = state.update_node_attrs("resources", new_resources)

# Edge attributes
new_state = state.update_edge_attrs("messages", new_messages)

# Adjacency matrices
new_state = state.update_adj_matrix("connections", new_adj)

# Global attributes
new_state = state.update_global_attr("round", state.global_attrs["round"] + 1)

# Full replacement
new_state = state.replace(node_types=new_types, ...)
```

## Agent System

### Flow

```
GraphState
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

### Key Functions

```python
# Extract agent's view of the world
def get_observation(state: GraphState, agent_id: int) -> ObservationMatrix:
    """
    Returns flat array with:
    - My node attributes
    - All node attributes (for neighbors)
    - Edge attributes to/from me
    - Adjacency (who I'm connected to)
    """
    ...

# Apply agent's action to state
def apply_action(state: GraphState, agent_id: int, action: ActionMatrix) -> GraphState:
    """
    Currently implements TRANSFER semantics:
    - Agents send resources to other agents
    - Zero-sum between agents
    - Can be swapped for EXTRACTION semantics
    """
    ...
```

### Policy Interface

```python
class Policy(Protocol):
    """Pure function: observation → action"""
    def __call__(self, obs: ObservationMatrix, key: PRNGKey) -> ActionMatrix:
        ...
```

**Key insight:** This is a Protocol, not a base class. Any callable with this signature works.

## JAX Integration Patterns

### Pattern 1: Factory for Static Shapes

JAX's JIT requires static shapes. Use factory functions:

```python
def make_run_simulation(n_rounds: int):
    """Create JIT-compiled simulation with static n_rounds."""

    @jit
    def run_single_simulation(agent_types, initial_state, key):
        # n_rounds is captured from closure (static)
        (final_state, _), _ = jax.lax.scan(
            step_fn,
            (initial_state, ...),
            None,
            length=n_rounds  # Static!
        )
        return final_state

    return run_single_simulation
```

### Pattern 2: vmap for Parallelization

Vectorize over batch dimension (usually keys):

```python
# Single simulation
run_sim = make_run_simulation(n_rounds=100)

# Vectorized over keys (batch dimension)
run_vec = jit(vmap(run_sim, in_axes=(None, None, 0)))
#                                           ^ vmap over keys

# Run batch of simulations in parallel
keys = random.split(key, batch_size)
results = run_vec(agent_types, initial_state, keys)
```

### Pattern 3: scan for Efficient Loops

Use `jax.lax.scan` instead of Python loops:

```python
def step_fn(carry, _):
    state, resources = carry
    new_state = update(state)
    new_resources = extract(new_state)
    return (new_state, new_resources), None

# Run n_rounds efficiently
(final_state, final_resources), _ = jax.lax.scan(
    step_fn,
    (initial_state, initial_resources),
    None,
    length=n_rounds
)
```

### Pattern 4: PRNGKey Threading

JAX requires explicit randomness threading:

```python
# Split key for multiple random operations
key, key1, key2 = random.split(key, 3)

action1 = policy1(obs1, key1)
action2 = policy2(obs2, key2)

# For multiple agents, split into array
keys = random.split(key, n_agents)
actions = vmap(policy, in_axes=(0, 0))(observations, keys)
```

## Simulation Loop

### Sequential (Baseline)

```python
state = initialize_state(...)

for round_idx in range(n_rounds):
    for agent_id in range(n_agents):
        # Observe
        obs = get_observation(state, agent_id)

        # Act
        action = policy(obs, key)

        # Update
        state = apply_action(state, agent_id, action)

    # Environment dynamics (regeneration, etc.)
    state = environment_step(state)
```

### Parallelized (vmap)

```python
# Vectorize everything
def step_all_agents(state, keys):
    """One round for all agents."""
    observations = vmap(get_observation, in_axes=(None, 0))(
        state, jnp.arange(n_agents)
    )

    actions = vmap(policy, in_axes=(0, 0))(observations, keys)

    # Apply actions (vmap or scan)
    for agent_id in range(n_agents):
        state = apply_action(state, agent_id, actions[agent_id])

    return state

# Run multiple rounds
def run_simulation(initial_state, key):
    keys = random.split(key, n_rounds)

    def scan_fn(state, round_key):
        agent_keys = random.split(round_key, n_agents)
        new_state = step_all_agents(state, agent_keys)
        new_state = environment_step(new_state)
        return new_state, None

    final_state, _ = jax.lax.scan(scan_fn, initial_state, keys)
    return final_state
```

## Lake Model Architecture

### Components

```
Lake Node (n_agents fish populations)
    ↑ regeneration
    ↓ extraction
Agents (n_agents cumulative resources)
```

### Dynamics

```python
# 1. Agents decide extraction
extractions = [agent_policy(lake_fish) for agent in agents]

# 2. Apply extractions (with scaling if over-limit)
total_extraction = sum(extractions)
if total_extraction > lake_fish:
    scale = lake_fish / total_extraction
    actual_extractions = [e * scale for e in extractions]
else:
    actual_extractions = extractions

# 3. Update lake
lake_fish = lake_fish - sum(actual_extractions)

# 4. Regenerate
growth = regeneration_rate * lake_fish * (1 - lake_fish / carrying_capacity)
lake_fish = lake_fish + growth

# 5. Update agent resources
agent_resources += actual_extractions
```

### Why it Works

- **Sustainable agents** extract fixed amounts (5 fish)
- **Exploiter agents** extract proportional (15% of lake)
- Initially: regeneration (80) > extraction (50) → sustainable
- As population drops: regeneration drops → eventually unsustainable
- Exploiters accelerate collapse (150 vs 5 fish per agent)

## Democratic Mechanisms

### PDD (Predictive Direct Democracy)

```python
def direct_democracy_vote(votes: jnp.ndarray) -> int:
    """All agents vote equally."""
    total_votes = jnp.sum(votes, axis=0)  # (n_portfolios,)
    return jnp.argmax(total_votes)
```

### PRD (Predictive Representative Democracy)

```python
def representative_democracy_vote(
    votes: jnp.ndarray,
    representatives: jnp.ndarray
) -> int:
    """Only representatives' votes count."""
    rep_votes = votes[representatives]
    total_votes = jnp.sum(rep_votes, axis=0)
    return jnp.argmax(total_votes)
```

### PLD (Predictive Liquid Democracy)

```python
def liquid_democracy_vote(
    votes: jnp.ndarray,
    delegations: jnp.ndarray,
    performance_history: jnp.ndarray
) -> int:
    """Performance-based delegation."""
    # Calculate vote weights based on delegations
    vote_weights = jnp.ones(len(votes))

    for i in range(len(delegations)):
        if delegations[i] >= 0:  # Is delegating
            delegate_id = delegations[i]
            vote_weights = vote_weights.at[delegate_id].add(vote_weights[i])
            vote_weights = vote_weights.at[i].set(0)

    # Weight votes by delegation power
    weighted_votes = votes * vote_weights[:, None]
    return jnp.argmax(jnp.sum(weighted_votes, axis=0))
```

## Performance Characteristics

### CPU (Current)
- Sequential: 0.075s for 4 simulations
- Vectorized (vmap): 0.062s for 4 simulations
- **Speedup: 1.3x**

### GPU (Expected in WSL)
- Same code, zero changes
- **Expected: 10-100x speedup**
- Larger batches (64-256) for better utilization

### Scaling Properties
- **Batch size:** Linear scaling up to hardware limits
- **Agent count:** Linear in most operations, quadratic in pairwise interactions
- **Rounds:** Linear (using scan)
- **Complexity per agent:** Directly impacts speedup gains

## Extension Points

### 1. New Agent Policies

```python
class MyPolicy:
    def __call__(self, obs: jnp.ndarray, key: PRNGKey) -> jnp.ndarray:
        # Your logic here
        return action
```

### 2. New Action Semantics

Modify `apply_action()` to change what actions mean:
- Current: Transfers between agents
- Alternative: Extraction from environment
- Alternative: State changes (beliefs, strategies)

### 3. New Environment Dynamics

Add to simulation loop:
```python
def environment_step(state: GraphState) -> GraphState:
    # Regeneration
    # Noise injection
    # Network rewiring
    # etc.
    return new_state
```

### 4. Learned Policies

```python
import optax

policy = LinearPolicy(input_dim, output_shape)
optimizer = optax.adam(learning_rate=0.01)

# Training loop with gradient descent
for epoch in range(n_epochs):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

## Testing Strategy

### Unit Tests
- Individual transformations
- Policy behavior
- State updates

### Integration Tests
- Full simulation runs
- Result verification
- Determinism checks

### Performance Tests
- CPU vs GPU speedup
- Batch size scaling
- JIT compilation time

### Validation Tests
- Baseline: Perfect information
- Debug: Simplified environments
- Production: Full complexity

---

**Next:** See NEXT_STEPS.md for development roadmap.
