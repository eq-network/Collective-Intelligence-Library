# Design Patterns and SOLID Principles

This document maps software engineering principles to the Collective Intelligence Library codebase.

---

## SOLID Principles Analysis

### Open/Closed Principle (OCP)

> "Software entities should be open for extension but closed for modification."

#### ✅ Good: Transform System

The Transform type alias allows unlimited extension without modifying core code:

```python
# From core/category.py:14
Transform = Callable[[GraphState], GraphState]
```

Anyone can create new transforms:

```python
def my_custom_transform(state: GraphState) -> GraphState:
    # Custom logic here
    return state.update_node_attrs("my_attr", new_values)

# Composes with existing transforms - no modification needed
pipeline = sequential(existing_transform, my_custom_transform)
```

#### ✅ Good: Strategy Pattern in Message Passing

From `engine/transformations/bottom_up/message_passing.py`:

```python
def create_message_passing_transform(
    connection_type: str,
    message_generator: MessageGenerator,    # ← Inject strategy
    message_processor: MessageProcessor     # ← Inject strategy
) -> Transform:
```

New message behaviors can be added by providing different generators/processors.

#### ⚠️ Gap: Duplicate Agent ABCs

Two different Agent base classes exist:

| Location | Signature | Issue |
|----------|-----------|-------|
| `core/agents.py:16` | `__init__(agent_id: int)` | Requires ID in constructor |
| `engine/agents/base.py:16` | No `__init__` | Simpler, ID passed to `__call__` |

**Recommendation**: Consolidate to the simpler `engine/agents/base.py` version.

---

### Dependency Inversion Principle (DIP)

> "High-level modules should not depend on low-level modules. Both should depend on abstractions."

#### ✅ Good: Transforms Depend on GraphState Interface

All transforms depend on the `GraphState` abstraction, not specific implementations:

```python
# High-level (transform) depends on abstraction (GraphState)
def market_clearing(state: GraphState) -> GraphState:
    prices = state.node_attrs["prices"]        # Interface method
    return state.update_node_attrs(...)        # Interface method
```

#### ⚠️ Gap: Environment Holds Mutable State

From `core/environment.py:63`:

```python
def step(self) -> Tuple[GraphState, bool]:
    # ...
    self.state = new_state  # ← MUTATION!
```

This violates DIP because the Environment directly manages state instead of delegating to transforms.

**Current**:
```
Environment ──────► Mutable self.state
     │
     └──► Transform (but then mutates self.state)
```

**Recommended**:
```
Environment ──────► Transform (pure)
     │                   │
     │                   ▼
     └────────────► Returns new GraphState
```

---

### Interface Segregation Principle (ISP)

> "No client should be forced to depend on methods it does not use."

#### ⚠️ Gap: Fat Environment Interface

From `core/environment.py`, the Environment ABC requires implementing all of:

```python
class Environment(ABC):
    @abstractmethod
    def get_observation_for_agent(self, agent: Agent): ...  # Observation

    @abstractmethod
    def apply_actions(self, actions: List[Action]): ...     # Action processing

    @abstractmethod
    def is_terminated(self) -> bool: ...                    # Termination

    @abstractmethod
    def reset(self) -> None: ...                            # Reset
```

A simple simulation that doesn't need observations still must implement `get_observation_for_agent`.

**Recommended Segregation**:

```python
class Steppable(Protocol):
    def step(self) -> Tuple[GraphState, bool]: ...

class Observable(Protocol):
    def get_observation_for_agent(self, agent: Agent) -> Dict[str, Any]: ...

class Resettable(Protocol):
    def reset(self) -> None: ...

class Terminable(Protocol):
    def is_terminated(self) -> bool: ...
```

---

### Liskov Substitution Principle (LSP)

> "Objects of a superclass should be replaceable with objects of its subclasses without affecting correctness."

#### ⚠️ Gap: ConservesSum Doesn't Actually Verify Conservation

From `core/property.py:67-75`:

```python
class ConservesSum(Property):
    def check(self, state: GraphState) -> bool:
        # BUG: Doesn't compare against reference!
        attr = state.node_attrs[self.attribute_name]
        return bool(jnp.all(jnp.isfinite(attr)))  # Just checks finite values
```

A transform could halve all resources and `ConservesSum` would still return `True`.

**Recommended Fix**:

```python
class ConservesSum(Property):
    def __init__(self, attribute_name: str, reference_sum: float = None):
        self.attribute_name = attribute_name
        self.reference_sum = reference_sum

    def bind(self, initial_state: GraphState) -> 'ConservesSum':
        """Create property bound to initial state's sum."""
        total = float(jnp.sum(initial_state.node_attrs[self.attribute_name]))
        return ConservesSum(self.attribute_name, reference_sum=total)

    def check(self, state: GraphState) -> bool:
        if self.reference_sum is None:
            raise ValueError("Property not bound to reference state")
        current_sum = float(jnp.sum(state.node_attrs[self.attribute_name]))
        return jnp.isclose(current_sum, self.reference_sum)
```

---

## Design Patterns

### Strategy Pattern ✅

**Where Used**: Message passing, market mechanisms, agent policies

```python
# Strategy interface
MessageGenerator = Callable[[GraphState, int], Message]
MessageProcessor = Callable[[GraphState, int, List[Message]], Dict[str, Any]]

# Context uses strategies
def create_message_passing_transform(
    message_generator: MessageGenerator,  # Strategy 1
    message_processor: MessageProcessor   # Strategy 2
) -> Transform:
```

**Benefits**:
- Swap market mechanisms without changing infrastructure
- Test different agent policies in isolation
- Compose behaviors from simple building blocks

---

### Decorator Pattern ❌ (Opportunity)

**Not Currently Used**, but would be valuable for:

```python
# Hypothetical decorator for transforms
def logging_decorator(transform: Transform) -> Transform:
    def decorated(state: GraphState) -> GraphState:
        print(f"Before: {state.global_attrs.get('round', 0)}")
        result = transform(state)
        print(f"After: {result.global_attrs.get('round', 0)}")
        return result
    return decorated

def validation_decorator(property: Property) -> Callable[[Transform], Transform]:
    def decorator(transform: Transform) -> Transform:
        def decorated(state: GraphState) -> GraphState:
            result = transform(state)
            assert property.check(result), f"Property {property.name} violated!"
            return result
        return decorated
    return decorator

# Usage
@validation_decorator(ConservesSum("resources"))
@logging_decorator
def my_transform(state: GraphState) -> GraphState:
    ...
```

---

### Factory Pattern ✅

**Where Used**: Transform creation functions

```python
# From message_passing.py
def create_message_passing_transform(
    connection_type: str,
    message_generator: MessageGenerator,
    message_processor: MessageProcessor
) -> Transform:
    """Factory that creates configured transform."""
    ...
```

**Benefits**:
- Encapsulates complex transform setup
- Parameters configure behavior at creation time
- Returned transform is simple `GraphState → GraphState`

---

## Summary Table

| Principle/Pattern | Status | Location | Notes |
|------------------|--------|----------|-------|
| OCP | ✅ Good | `core/category.py` | Transform composition works well |
| DIP | ⚠️ Partial | `core/environment.py` | Environment mutates state |
| ISP | ⚠️ Violated | `core/environment.py` | Fat Environment interface |
| LSP | ⚠️ Unclear | `core/property.py` | ConservesSum doesn't verify |
| Strategy | ✅ Good | `message_passing.py` | Clean strategy injection |
| Decorator | ❌ Missing | - | Opportunity for transform wrappers |
| Factory | ✅ Good | `message_passing.py` | `create_*_transform()` pattern |

---

## Recommended Actions

### High Priority

1. **Consolidate Agent ABCs**: Choose one (recommend `engine/agents/base.py`)
2. **Fix ConservesSum**: Add reference binding to actually verify conservation

### Medium Priority

3. **Segregate Environment interface**: Break into Steppable, Observable, etc.
4. **Make Environment stateless**: Pass state explicitly, don't store in `self.state`

### Low Priority (Enhancements)

5. **Add Decorator pattern**: For logging, validation, timing of transforms
6. **Document Property contracts**: Make LSP guarantees explicit

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - How patterns fit in the layers
- [COMPUTATIONAL_FOUNDATION.md](./COMPUTATIONAL_FOUNDATION.md) - Why immutability matters
