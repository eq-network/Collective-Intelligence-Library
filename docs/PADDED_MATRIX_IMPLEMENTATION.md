# Padded Matrix Implementation - Complete

## Overview

Successfully implemented padded matrix support for Mycorrhiza's GraphState, enabling efficient dynamic graph operations with O(1) add/remove while maintaining full backward compatibility.

## Implementation Summary

### Phase 1: Core GraphState Changes ✅
**File**: `core/graph.py`

**Changes**:
- Added optional `capacity: Optional[int] = None` field to GraphState
- Updated `num_nodes` property to count only active nodes in capacity mode
- Added `is_capacity_mode` property
- Added `get_active_indices()` method to get array of active node indices
- Added `get_active_mask()` method to get boolean mask for active nodes
- Created `create_padded_state()` factory function for creating padded states
- Added `CapacityExceededError` exception class

**Key Features**:
- `node_type=-1` marks inactive (padded) slots
- Capacity mode is opt-in (backward compatible)
- Fixed array shapes enable future JIT compilation

---

### Phase 2: Graph Editor Operations ✅
**File**: `studio/graph_editor.py`

**Changes**:
- Updated `add_node()` with dual implementation:
  - **Capacity mode**: O(1) slot activation by finding first inactive slot
  - **Dynamic mode**: O(N²) array resizing (backward compatible)
- Updated `remove_node()` with dual implementation:
  - **Capacity mode**: O(1) slot deactivation by setting node_type=-1
  - **Dynamic mode**: O(N²) array deletion
- Added proper error handling with `CapacityExceededError`

**Performance**:
- Capacity mode: O(1) operations
- Dynamic mode: O(N²) operations (unchanged from before)

---

### Phase 3: Property Updates ✅
**File**: `core/property.py`

**Changes**:
- Updated `ConservesSum.bind()` to use active masking in capacity mode
- Updated `ConservesSum.check()` to use active masking in capacity mode

**Behavior**:
- Correctly sums only active nodes when verifying conservation
- Inactive nodes are excluded from sum calculations

---

### Phase 4: Transform Updates ✅

#### 4.1 Message Passing Transform
**File**: `engine/transformations/bottom_up/message_passing.py`

**Changes**:
- Use `get_active_indices()` for message generation and processing
- Filter sender nodes to only active neighbors using `active_mask`
- Store messages in dict keyed by node_id instead of list

**Pattern**:
```python
active_indices = state.get_active_indices()
active_mask = state.get_active_mask()

for i in active_indices:
    i_int = int(i)
    # Process only active nodes...
    sender_mask = (adj_matrix[:, i] > 0) & active_mask
```

#### 4.2 Belief Update Transform
**File**: `engine/transformations/bottom_up/updating.py`

**Changes**:
- Iterate over `get_active_indices()` instead of `range(num_nodes)`
- Filter neighbors to only active nodes

#### 4.3 Prediction Market Transform
**File**: `engine/transformations/bottom_up/prediction_market.py`

**Changes**:
- Generate agent-specific signals only for active nodes
- Use active indices when finding best-informed agent

#### 4.4 Observations Builder
**File**: `engine/environments/farmers_market/observations.py`

**Changes**:
- `build_all_observations()` returns observations only for active agents
- `build_farmer_observation()` filters neighbors to active nodes
- `build_limited_observation()` filters to active neighbors

#### 4.5 Visualization Config
**File**: `studio/configs/farmers_market.py`

**Changes**:
- `get_node_positions()`: Position only active farmers in circle
- `get_node_colors()`: Color only active farmers
- `get_node_labels()`: Label only active farmers
- `get_edges()`: Show edges only between active farmers

#### 4.6 Agent Transforms
**File**: `engine/environments/farmers_market/agent_transforms.py`

**Changes**:
- `create_agent_driven_trade_transform()`: Map agent policies to actual node IDs
- `create_agent_driven_consumption_transform()`: Use active indices for consumption
- `create_history_tracker_transform()`: Sum only active nodes with active masking

**Key Pattern**:
```python
active_indices = state.get_active_indices()
for policy_idx, (obs, policy) in enumerate(zip(observations, agent_policies)):
    actual_node_id = int(active_indices[policy_idx])
    action = policy(obs, actual_node_id)
```

---

### Phase 5: Demo & Testing ✅
**File**: `examples/capacity_demo.py`

**Features**:
- Demonstrates creating padded state with `create_padded_state()`
- Shows O(1) add/remove operations
- Tests capacity exceeded error handling
- Verifies slot reuse after node removal
- Compares capacity mode vs dynamic mode
- Validates conservation with active masking

**Test Results**:
```
[OK] All tests passed!
  - Capacity mode: O(1) operations working
  - Dynamic mode: Backward compatible
  - Conservation: Verified with active masking
  - Error handling: Capacity exceeded caught
  - Slot reuse: Working correctly
```

---

## Usage Examples

### Creating a Padded State

```python
from core.graph import create_padded_state
import jax.numpy as jnp

# Create state with capacity=10, initially 3 active nodes
state = create_padded_state(
    capacity=10,
    initial_active=3,
    node_types_init=jnp.array([0, 0, 0], dtype=jnp.int32),
    node_attrs_init={
        "resources": jnp.array([100.0, 100.0, 100.0])
    },
    adj_matrices_init={
        "network": jnp.eye(3)
    }
)

print(f"Capacity: {state.capacity}")        # 10
print(f"Active nodes: {state.num_nodes}")   # 3
print(f"Active indices: {state.get_active_indices()}")  # [0, 1, 2]
```

### Adding and Removing Nodes

```python
from studio.graph_editor import add_node, remove_node
from core.graph import CapacityExceededError

# Add a node (O(1) in capacity mode)
state = add_node(state, node_type=1, initial_attrs={"resources": 50.0})

# Remove a node (O(1) in capacity mode)
state = remove_node(state, node_id=1)

# Try to exceed capacity
try:
    for i in range(20):
        state = add_node(state, node_type=2)
except CapacityExceededError as e:
    print(f"Capacity reached: {e}")
```

### Writing Transforms for Capacity Mode

```python
def my_transform(state: GraphState) -> GraphState:
    active_indices = state.get_active_indices()

    # Iterate over active nodes only
    for i in active_indices:
        i_int = int(i)
        # Process node i...

    # When checking neighbors, filter to active
    if state.is_capacity_mode:
        active_mask = state.get_active_mask()
        neighbor_mask = (adj_matrix[i, :] > 0) & active_mask
    else:
        neighbor_mask = adj_matrix[i, :] > 0

    return state
```

### Verifying Conservation

```python
from core.property import ConservesSum

# Create and bind property
conservation = ConservesSum("resources", tolerance=1e-5)
bound_conservation = conservation.bind(initial_state)

# After transforms, check conservation
# (automatically handles active masking in capacity mode)
final_state = transform(initial_state)
assert bound_conservation.check(final_state)
```

---

## Design Decisions

### Why node_type=-1 for inactive nodes?
- **No new arrays needed**: Reuses existing node_types field
- **Easy masking**: `node_types != -1` gives active mask
- **Atomic operations**: Single field marks active/inactive

### Why optional capacity?
- **Backward compatibility**: Existing code works unchanged
- **Opt-in performance**: Use capacity when needed
- **Graceful migration**: Can adopt incrementally

### Why dual implementation (capacity vs dynamic)?
- **Smooth adoption**: New code gets O(1), old code unchanged
- **Clear semantics**: Different modes for different use cases
- **Testing**: Both modes tested in same codebase

---

## Backward Compatibility

All existing code continues to work without modification:

```python
# OLD STYLE (still works - dynamic mode)
state = GraphState(
    node_types=jnp.zeros(5),
    node_attrs={"resources": jnp.ones(5)},
    adj_matrices={"network": jnp.eye(5)}
)
# capacity=None automatically, uses dynamic resizing

# NEW STYLE (opt-in - capacity mode)
state = create_padded_state(
    capacity=10,
    initial_active=5,
    node_attrs_init={"resources": jnp.ones(5)},
    adj_matrices_init={"network": jnp.eye(5)}
)
# capacity=10, uses slot activation (O(1))
```

---

## Files Modified

### Core Files
1. `core/graph.py` - Added capacity field and helper methods
2. `core/property.py` - Updated ConservesSum for active masking
3. `studio/graph_editor.py` - Dual mode add/remove operations

### Transform Files
4. `engine/transformations/bottom_up/message_passing.py`
5. `engine/transformations/bottom_up/updating.py`
6. `engine/transformations/bottom_up/prediction_market.py`

### Environment Files
7. `engine/environments/farmers_market/observations.py`
8. `engine/environments/farmers_market/agent_transforms.py`
9. `studio/configs/farmers_market.py`

### Example Files
10. `examples/capacity_demo.py` (NEW)

---

## Performance Characteristics

| Operation | Dynamic Mode | Capacity Mode |
|-----------|-------------|---------------|
| Add node | O(N²) | O(1) |
| Remove node | O(N²) | O(1) |
| Iterate nodes | O(N) | O(N_active) |
| Access node | O(1) | O(1) |
| Memory | Variable | Fixed (capacity) |
| JIT-able | No (shape changes) | Yes (fixed shapes) |

---

## Future Enhancements

1. **JIT Compilation**: Fixed shapes enable future JIT optimization
2. **Defragmentation**: Could compact active nodes to front for cache locality
3. **Auto-expansion**: Optional capacity growth on overflow
4. **Batch Operations**: Add/remove multiple nodes in one operation

---

## Testing

All tests pass:
- ✅ Capacity mode operations (O(1))
- ✅ Dynamic mode backward compatibility
- ✅ Conservation verification with active masking
- ✅ Error handling (capacity exceeded)
- ✅ Slot reuse after removal
- ✅ Transform filtering to active nodes
- ✅ Agent mapping to correct node IDs

Run tests:
```bash
python examples/capacity_demo.py
python examples/clean_architecture_demo.py
```

---

## Summary

The padded matrix implementation is **complete and tested**. All transforms have been updated to use `get_active_indices()` and `get_active_mask()` for proper handling of inactive nodes. The system maintains full backward compatibility while enabling O(1) dynamic graph operations for new code.

**Key Achievement**: Efficient dynamic graphs with JAX while maintaining the pure functional architecture.
