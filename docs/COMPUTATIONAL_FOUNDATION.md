# Computational Foundation

This document explains why the Collective Intelligence Library is built on immutability, and how this enables parallelization and efficient computation.

## The Key Insight

**Immutability isn't just a coding style - it's a computational requirement for parallelization.**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   OOP Patterns           FP Purity           JAX Execution      │
│   (WHAT varies)          (HOW composed)      (WHERE runs)       │
│                                                                 │
│   ┌─────────────┐        ┌─────────────┐     ┌─────────────┐   │
│   │ Strategy    │        │ Transform   │     │ vmap        │   │
│   │ Pattern     │───────►│ Composition │────►│ pmap        │   │
│   │             │        │             │     │ jit         │   │
│   └─────────────┘        └─────────────┘     └─────────────┘   │
│         │                      │                   │            │
│   Enables:               Enables:            Enables:           │
│   - Swap mechanisms      - Verify purity     - GPU/TPU          │
│   - Test isolation       - Compose safely    - Parallelism      │
│   - External APIs        - Reason locally    - Scale to N       │
│                                                                 │
│           ┌─────────────────────────────────────┐              │
│           │     IMMUTABILITY BRIDGES ALL        │              │
│           │     THREE CONCERNS                  │              │
│           └─────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Mutation Prevents Parallelization

Consider a typical OOP approach:

```python
# MUTABLE (Traditional OOP)
class Agent:
    def __init__(self):
        self.resources = 100

    def update(self, amount):
        self.resources += amount  # MUTATION!

# Problem: Multiple threads accessing the same agent
Thread 1: agent.update(10)  ──┐
Thread 2: agent.read()      ───┼──► RACE CONDITION!
Thread 3: agent.update(-5)  ──┘
```

With mutation, you need locks, synchronization, and careful coordination. This doesn't scale.

---

## How Immutability Enables Parallelization

The Collective Intelligence Library approach:

```python
# IMMUTABLE (Functional)
@dataclass(frozen=True)
class GraphState:
    node_attrs: Dict[str, jnp.ndarray]  # frozen = immutable

    def update_node_attrs(self, name, values):
        # Returns NEW state, original unchanged
        return self.replace(node_attrs={**self.node_attrs, name: values})

# All threads can safely read the same state
Thread 1: transform1(state) → state_1  ─┐
Thread 2: transform2(state) → state_2  ─┼─► ALL READ SAME ORIGINAL
Thread 3: transform3(state) → state_3  ─┘

# Merge results (if needed)
final = combine(state_1, state_2, state_3)
```

**Key principle**: All reads happen from the original state. All writes create new state. No conflicts possible.

---

## Large Adjacency Matrices as the Substrate

The entire simulation state is represented as matrices:

```python
GraphState:
    node_types:    [N]         # Type label per node
    node_attrs:    {
        "resources": [N],      # Diagonal: node i's resources
        "beliefs":   [N],      # Diagonal: node i's beliefs
        ...
    }
    adj_matrices:  {
        "trade":     [N×N],    # Off-diagonal: i→j trade edges
        "trust":     [N×N],    # Off-diagonal: i→j trust edges
        ...
    }
```

Matrix operations are inherently parallelizable:

```python
# Update ALL nodes at once (vectorized)
new_resources = state.node_attrs["resources"] * 1.05  # 5% growth

# Message passing is matrix multiplication
messages = adj_matrix @ node_values  # All messages computed in parallel
```

---

## JAX Operations

JAX provides three key operations that leverage immutability:

### 1. `jax.jit` - Just-In-Time Compilation

```python
from core.category import jit_transform

# Compile the transform for faster execution
fast_transform = jit_transform(my_transform)

# First call compiles, subsequent calls are fast
result = fast_transform(state)
```

### 2. `jax.vmap` - Vectorization

```python
# Write logic for ONE node
def process_single_node(node_state):
    return node_state * 2

# Automatically applies to ALL nodes in parallel
process_all_nodes = jax.vmap(process_single_node)
results = process_all_nodes(state.node_attrs["resources"])
```

### 3. `jax.pmap` - Multi-device Parallelization

```python
# Distribute across GPUs/TPUs
parallel_transform = jax.pmap(transform)
```

---

## Annotated Example: Message Passing

From `engine/transformations/bottom_up/message_passing.py`:

```python
def create_message_passing_transform(
    connection_type: str,
    message_generator: MessageGenerator,    # Strategy pattern (OOP)
    message_processor: MessageProcessor     # Strategy pattern (OOP)
) -> Transform:                             # Returns pure function (FP)

    def transform(state: GraphState) -> GraphState:
        num_nodes = state.num_nodes
        adj_matrix = state.adj_matrices.get(connection_type)

        # ═══════════════════════════════════════════════════════════
        # Line 44: "Generate all messages in parallel (conceptually)"
        # ═══════════════════════════════════════════════════════════
        # Current: Python list comprehension (sequential)
        messages = [message_generator(state, i) for i in range(num_nodes)]

        # JAX-optimized would be:
        # messages = jax.vmap(lambda i: message_generator(state, i))(
        #     jnp.arange(num_nodes)
        # )
        # ═══════════════════════════════════════════════════════════

        # Process messages for each node
        new_node_attrs_updates = []
        for i in range(num_nodes):
            sender_indices = jnp.where(adj_matrix[:, i] > 0)[0]
            incoming_messages = [messages[j] for j in sender_indices]
            updates = message_processor(state, i, incoming_messages)
            new_node_attrs_updates.append(updates)

        # ═══════════════════════════════════════════════════════════
        # Lines 60-61: "All reads from original `state`, writes batched"
        # ═══════════════════════════════════════════════════════════
        # This is the key to purity:
        # - Every read: from `state` (the input)
        # - Every write: batched into `final_node_attrs` (the output)
        # - Original `state` is NEVER modified
        # ═══════════════════════════════════════════════════════════

        final_node_attrs = state.node_attrs.copy()
        for attr_name in final_node_attrs.keys():
            if any(attr_name in updates for updates in new_node_attrs_updates):
                new_values = final_node_attrs[attr_name].copy()
                for i, updates in enumerate(new_node_attrs_updates):
                    if attr_name in updates:
                        # JAX immutable update pattern
                        new_values = new_values.at[i].set(updates[attr_name])
                final_node_attrs[attr_name] = new_values

        return state.replace(node_attrs=final_node_attrs)

    return transform
```

---

## The `.at[].set()` Pattern

JAX arrays are immutable. You cannot do:

```python
# ✗ WRONG - This won't work with JAX
array[5] = 42
```

Instead, use the functional update pattern:

```python
# ✓ CORRECT - Returns new array
new_array = array.at[5].set(42)

# Original array is unchanged
print(array[5])      # Still the old value
print(new_array[5])  # 42
```

This is used throughout the codebase:

```python
# From graph.py:181
new_node_attrs[attr_name] = new_node_attrs[attr_name].at[node_id].set(new_value)
```

---

## Performance Hierarchy

When optimizing, follow this order:

1. **First**: Get the math right (pure transforms)
2. **Second**: Verify correctness (property checks)
3. **Third**: Optimize execution (JIT, vmap, pmap)

```python
# 1. Define pure transform
def my_transform(state: GraphState) -> GraphState:
    ...

# 2. Verify it works
assert validate_properties(my_transform, initial, final)

# 3. Then optimize
fast_transform = jit_transform(my_transform)
```

---

## Summary

| Concept | Enables |
|---------|---------|
| Immutability | Safe parallel reads, no race conditions |
| Matrix representation | Vectorized operations on all nodes |
| `jax.jit` | Compiled, fast execution |
| `jax.vmap` | Single-node logic → all-node parallel |
| `jax.pmap` | Multi-device distribution |
| `.at[].set()` | Functional array updates |

The combination of immutable data structures and pure transforms allows the same code to run on 10 nodes or 10,000 nodes - the complexity is in the data, not the code.

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - How layers connect
- [DESIGN_PATTERNS.md](./DESIGN_PATTERNS.md) - OOP patterns that work with immutability
