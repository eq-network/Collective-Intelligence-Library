# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Continuity

**IMPORTANT:** Before starting work, check these files for context from previous sessions:
- `.claude/CURRENT_PLAN.md` - Active plans and next steps
- `.claude/SESSION_LOG.md` - History of decisions and progress

At the end of a session, offer to update these files with current progress.

## What This Is

**Mycorrhiza** is a programming language for collective intelligence. Not a simulation framework, not a multi-agent library—a language where the fundamental unit of computation is a **transformation on graphs**.

Think of it like:
- **SQL** is a language for querying data
- **React** is a language for describing UIs as functions of state
- **Mycorrhiza** is a language for describing coordination mechanisms as compositions of transformations

**Core principle**: `GraphState → Transform → GraphState → Transform → ...`

## The Language Mental Model

### Transformations as First-Class Citizens

Everything is a transformation:

```python
# A transformation is a pure function: GraphState → GraphState
type Transform = Callable[[GraphState], GraphState]

# Markets are transformations
market_transform: Transform

# Networks are transformations
network_transform: Transform

# Democracy is a transformation
democracy_transform: Transform

# Composition is the primary operation
hybrid_system = compose(
    network_transform,
    market_transform,
    democracy_transform
)
```

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│ EMERGENCE LAYER                                  │
│ Markets, Networks, Democracies emerge here       │
│ (Not primitives - they're patterns!)            │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│ PROTOCOL LAYER                                   │
│ Transform functions define local computation     │
│ - price_update(state, order) → new_state       │
│ - belief_update(state, info) → new_state       │
│ - vote_aggregate(state, votes) → new_state     │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│ SUBSTRATE LAYER                                  │
│ Pure message-passing on graphs                   │
│ - Asynchronous, distributed, parallel           │
│ - No semantic assumptions                        │
└──────────────────────────────────────────────────┘
```

**The Router Insight**: Just like a router doesn't know about "video" or "email"—it just forwards packets—the substrate doesn't know about "markets" or "democracy"—it just applies transforms to messages.

## Design Principles

### 1. Purity Above All

Every transformation must be a pure function. No side effects. No mutation. State → State.

```python
# ✓ CORRECT: Pure transformation
def information_diffusion(state: GraphState) -> GraphState:
    return state.update_node_attrs(
        "beliefs",
        diffuse_on_network(state.node_attrs["beliefs"], state.adj_matrices["network"])
    )

# ✗ WRONG: Mutation
def information_diffusion(state: GraphState) -> None:
    state.node_attrs["beliefs"] = ...  # NEVER DO THIS!
```

### 2. Types Encode Mathematical Properties

Use the type system to guarantee invariants:

```python
# This type guarantees voting power is conserved
@conserves_sum("voting_power")
def delegation_transform(state: GraphState) -> GraphState:
    ...

# This type guarantees convergence
@converges_to_fixed_point(tolerance=0.01)
def belief_diffusion(state: GraphState) -> GraphState:
    ...
```

### 3. Composition is Fundamental

The language's power comes from composing simple transforms into complex behaviors:

```python
# Sequential composition
pld_mechanism = sequential(
    information_sharing,
    delegation_update,
    voting_power_calculation,
    prediction_market,
    voting_aggregation
)

# Parallel composition
multi_protocol = parallel(
    market_protocol,
    network_protocol,
    democracy_protocol
)

# Conditional composition
adaptive_system = conditional(
    market_protocol,
    predicate=lambda state: state.global_attrs["market_active"],
    otherwise=auction_protocol
)
```

### 4. Separate Process from Execution

**NEVER mix what a transformation does with how it executes:**

```python
# Process layer: WHAT happens (pure math)
diffusion = DiffusionTransform(
    attribute="information",
    method=WeightedAverage(),
    preserves=[ConservesTotal(), Converges()]
)

# Execution layer: HOW it's computed (separate concern)
result = executor.apply(
    diffusion,
    state,
    strategy=ParallelStrategy(threads=8)  # Execution detail
)
```

## Language Primitives

### GraphState: The Universal Type

All transformations operate on `GraphState` (defined in `core/graph.py`):

```python
@dataclass(frozen=True)
class GraphState:
    node_types: jnp.ndarray              # Node type labels
    node_attrs: Dict[str, jnp.ndarray]   # Per-node attributes
    adj_matrices: Dict[str, jnp.ndarray] # Graph connectivity
    global_attrs: Dict[str, Any]         # System-level state
```

**Critical**: GraphState is **immutable**. Every transformation returns a new GraphState.

Update methods:
- `state.update_node_attrs(attr_name, new_values) → GraphState`
- `state.update_adj_matrix(rel_name, new_matrix) → GraphState`
- `state.update_global_attr(attr_name, value) → GraphState`

### Transform: The Core Abstraction

All transforms follow this signature (from `core/category.py`):

```python
def transform(
    state: GraphState,
    random_key: Optional[RandomKey] = None
) -> GraphState:
    """
    Pure transformation of graph state.

    Args:
        state: Current system state
        random_key: Optional randomness (JAX style)

    Returns:
        New system state with transformation applied
    """
```

### Composition Operators

From `core/category.py`:
- **`sequential(*transforms)`**: Chains transforms in order (f₁ → f₂ → f₃)
- **`compose(f, g)`**: Composes two transforms (f followed by g)
- **`identity()`**: Returns the no-op transform
- **`attach_properties(transform, {props})`**: Annotates mathematical guarantees

### Properties: Mathematical Invariants

From `core/property.py`:
- **`ConservesSum(attribute_name)`**: Ensures total sum remains constant
- Properties compose via: `&` (and), `|` (or), `~` (not)
- Attach to transforms to document/verify guarantees

### Messages

Messages are structured data flowing on the graph:

```python
@dataclass
class Message:
    sender: NodeId
    receiver: NodeId
    content: Any
    protocol: str  # "market", "network", "vote", etc.
```

Different protocols interpret the same substrate differently.

## Repository Structure

```
mycorrhiza/
├── core/                    # Language primitives
│   ├── graph.py            # GraphState definition
│   ├── category.py         # Transform composition
│   ├── property.py         # Mathematical invariants
│   ├── simulation.py       # Legacy OOP wrapper
│   ├── environment.py      # Legacy OOP wrapper
│   └── agents.py           # Legacy OOP wrapper
├── engine/                  # Protocol library
│   ├── transformations/
│   │   ├── bottom_up/      # Agent-level transforms (emergent)
│   │   │   ├── message_passing.py
│   │   │   ├── updating.py
│   │   │   ├── token_economy.py
│   │   │   └── prediction_market.py
│   │   └── top_down/       # System-level transforms (imposed)
│   │       ├── resource.py
│   │       └── market.py
│   └── agents/             # Example agent implementations
├── execution/               # Execution strategies (HOW, not WHAT)
│   ├── config.py           # Experiment configuration
│   └── worker.py           # Parallel execution worker
├── services/                # Optional integrations
│   └── llm.py              # LLM service for agents
└── requirements.txt
```

### Two Architectural Patterns (Important!)

The codebase has two patterns that coexist:

1. **Functional Core (Primary)**: The pure functional transformation language
   - Location: `core/graph.py`, `core/category.py`, `engine/transformations/`
   - Pattern: Pure functions `GraphState → GraphState`
   - **Use this for all new work**

2. **OOP Wrapper (Legacy)**: Traditional agent-based simulation classes
   - Location: `core/simulation.py`, `core/environment.py`, `core/agents.py`
   - Pattern: Agent/Environment classes with step() methods
   - Only exists for backwards compatibility and experiment execution

**Always prefer the functional approach.** The OOP wrapper is legacy code.

## Working with the Language

### Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, JAX, NumPy, pandas

### Building a New Protocol

Think in terms of language design:

```python
# 1. Define the protocol as a transform factory
def create_price_discovery_transform(
    price_sensitivity: float,
    convergence_rate: float
) -> Transform:
    """
    Creates a price discovery transformation.

    Properties preserved:
    - Resource conservation
    - No arbitrage opportunities
    """
    def transform(state: GraphState) -> GraphState:
        # Read current state
        offers = state.node_attrs["offers"]
        prices = state.global_attrs["prices"]

        # Compute new prices (local computation)
        new_prices = market_clearing_logic(offers, prices, price_sensitivity)

        # Return new state
        return state.update_global_attr("prices", new_prices)

    # Annotate properties
    return attach_properties(transform, {
        ConservesSum("resources"),
        NoArbitrage()
    })

# 2. Compose with other protocols
coordination_mechanism = sequential(
    network_propagation,
    create_price_discovery_transform(sensitivity=0.5, rate=0.1),
    democratic_regulation
)

# 3. Execute
final_state = coordination_mechanism(initial_state)
```

### Message Passing Pattern

Use `engine/transformations/bottom_up/message_passing.py` as a template:

```python
from engine.transformations.bottom_up.message_passing import create_message_passing_transform

my_protocol = create_message_passing_transform(
    connection_type="network_name",  # Which adjacency matrix to use
    message_generator=lambda state, sender_id: {
        "content": state.node_attrs["belief"][sender_id],
        "confidence": state.node_attrs["confidence"][sender_id]
    },
    message_processor=lambda state, receiver_id, messages: {
        "belief": aggregate_beliefs(messages),
        "confidence": update_confidence(messages)
    }
)
```

### Handling Randomness

Follow JAX's functional random key approach:

```python
import jax.random as jr

def stochastic_transform(state: GraphState, key: jr.PRNGKey) -> GraphState:
    # Split key for independent random operations
    key1, key2 = jr.split(key)

    noise = jr.normal(key1, shape=(state.num_nodes,))
    mutation = jr.bernoulli(key2, p=0.1, shape=(state.num_nodes,))

    new_values = state.node_attrs["value"] + noise * mutation
    return state.update_node_attrs("value", new_values)

# Use it:
key = jr.PRNGKey(42)
final_state = stochastic_transform(initial_state, key)
```

### Performance Optimization

**Order of operations**:

1. **FIRST**: Get the math right (pure transforms)
2. **THEN**: Optimize execution separately

```python
from core.category import jit_transform

# Optimize execution (doesn't change semantics)
fast_transform = jit_transform(my_transform)
```

Use JAX JIT, vmap, pmap for acceleration—but only after verifying correctness.

## Development Commands

### No Build System

There is no build system, Makefile, or test suite currently. Run Python files directly:

```bash
python your_script.py
```

### Testing Pattern

When adding features, write validation functions:

```python
def validate_conservation(initial: GraphState, final: GraphState, attr: str):
    """Verify that total sum is conserved."""
    initial_sum = jnp.sum(initial.node_attrs[attr])
    final_sum = jnp.sum(final.node_attrs[attr])
    assert jnp.isclose(initial_sum, final_sum), f"Conservation violated: {initial_sum} → {final_sum}"

# Use in development
initial_state = create_state()
final_state = my_transform(initial_state)
validate_conservation(initial_state, final_state, "resources")
```

## Common Patterns

### Conditional Transforms

```python
def conditional_transform(condition_fn):
    def transform(state):
        if condition_fn(state):
            return some_transform(state)
        return state
    return transform
```

### Parameterized Transform Factories

```python
def create_diffusion_transform(rate: float, attribute: str) -> Transform:
    def diffusion_transform(state):
        values = state.node_attrs[attribute]
        network = state.adj_matrices["connections"]

        # Diffusion logic using rate
        new_values = values + rate * (network @ values - values)

        return state.update_node_attrs(attribute, new_values)
    return diffusion_transform

# Use it:
belief_diffusion = create_diffusion_transform(rate=0.1, attribute="beliefs")
resource_diffusion = create_diffusion_transform(rate=0.05, attribute="resources")
```

### Debug Transforms

Insert in pipeline to inspect state:

```python
def debug_transform(label: str = ""):
    def transform(state):
        print(f"--- DEBUG {label} ---")
        print(f"Round: {state.global_attrs.get('round', 'N/A')}")
        print(f"Node attrs: {list(state.node_attrs.keys())}")
        print(f"First 5 of 'score': {state.node_attrs.get('score', [])[:5]}")
        return state
    return transform

pipeline = sequential(
    transform1,
    debug_transform("After transform1"),
    transform2,
    debug_transform("After transform2")
)
```

## Key Implementation Guidelines

### Prefer Functional Composition Over Classes

```python
# ✓ GOOD: Functional composition
def compose(*transforms):
    def composed(state):
        for transform in transforms:
            state = transform(state)
        return state
    return composed

# ✗ AVOID: Class-based composition
class ComposedTransform:
    def __init__(self, *transforms):
        self.transforms = transforms
    def apply(self, state):
        ...
```

### Prefer Explicit Data Flow

```python
# ✓ GOOD: Explicit data flow
new_state = transform1(transform2(transform3(initial_state)))

# Or using sequential:
pipeline = sequential(transform3, transform2, transform1)
final_state = pipeline(initial_state)

# ✗ AVOID: Implicit state management
system.apply(transform1)
system.apply(transform2)
system.apply(transform3)
```

### JAX Array Operations

JAX arrays are immutable. Use `.at[].set()` for updates:

```python
# ✓ CORRECT: Immutable update
new_array = old_array.at[5].set(42)
new_array = old_array.at[indices].set(values)

# ✗ WRONG: Direct assignment (won't work with JAX)
old_array[5] = 42
```

## Think Like a Language Designer

When someone asks for a feature, ask:

1. **Is this a new primitive or a composition of existing ones?**
   - If composition → show them how to compose existing transforms
   - If primitive → carefully consider if it belongs in the core

2. **What properties does it need to preserve?**
   - Conservation laws? Convergence? Monotonicity?
   - Encode these as Property annotations

3. **How does it compose with other transforms?**
   - Test sequential composition
   - Test parallel composition
   - Verify properties are preserved through composition

4. **Can it be expressed in the existing language?**
   - Don't add new features if they can be composed from existing ones

## Not Everything Needs to Be Built In

Like Lisp has few primitives but enormous expressiveness:

- **Small core**: Message-passing, transforms, composition
- **Rich library**: Protocol implementations built from core
- **User extensions**: New protocols composable with existing

## Reference Documentation

- **`README.md`**: High-level overview and quick examples
- **`Start_Here.md`**: 30-minute tutorial for building first simulation
- **`Manifesto.md`**: Design philosophy and theoretical foundations

## Current Status and Future Direction

The codebase is implementing this language vision but isn't complete. When contributing:

1. **Reinforce core principles**: Purity, composition, separation of concerns
2. **Push toward language-like abstractions**: How would this look if it were a real programming language?
3. **Question deviations**: If something breaks purity or forces mutation, suggest alternatives
4. **Think about composition**: Every new feature should compose cleanly with existing ones

## Final Note

This isn't just a simulation tool. It's a formal language for reasoning about collective intelligence—where:
- Types enforce mathematical properties
- Composition rules prevent emergent failures
- Execution model enables efficient computation

Every design decision should ask: **"Does this make the language more expressive, more compositional, more verifiable?"**

Think like you're building the collective intelligence equivalent of SQL, React, or TensorFlow—a domain-specific language that fundamentally changes how people build coordination systems.
