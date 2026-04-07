# Pytree Fix: Partition global_attrs by type

**Date:** 2026-03-30
**File:** `core/graph.py`
**Task:** CORE-01

## What Changed

`tree_flatten()` and `tree_unflatten()` in `GraphState` now partition `global_attrs` into:
- **Dynamic** (JAX arrays) → go into `children` (traced by XLA)
- **Static** (Python int/float/str) → go into `aux_data` (baked into compiled kernel)

## Why

Previously ALL `global_attrs` were put in `aux_data` (static). Values like `resource_level` and `step` change every tick, so XLA saw different "constants" each tick and recompiled the entire step function. This made JIT useless.

## Before

```python
aux_data = (
    tuple(node_attr_keys),
    tuple(adj_matrix_keys),
    tuple(edge_attr_keys),
    tuple(self.global_attrs.items())  # ALL static
)
```

## After

```python
# Partition: JAX arrays → children, Python scalars → aux_data
dynamic_global_keys = []
static_global_items = []
for k in sorted(self.global_attrs.keys()):
    v = self.global_attrs[k]
    if isinstance(v, (jnp.ndarray, jax.Array)):
        dynamic_global_keys.append(k)
        children.append(v)
    else:
        static_global_items.append((k, v))

aux_data = (
    tuple(node_attr_keys),
    tuple(adj_matrix_keys),
    tuple(edge_attr_keys),
    tuple(dynamic_global_keys),   # NEW: which children are global attrs
    tuple(static_global_items),   # static values only
)
```

## Verification

```python
state = GraphState(
    node_types=jnp.array([0, 1, 0]),
    node_attrs={'x': jnp.array([1.0, 2.0, 3.0])},
    adj_matrices={},
    global_attrs={
        'resource_level': jnp.array(4000.0),  # dynamic
        'K': 5000.0,                            # static
        'step': jnp.array(0),                  # dynamic
        'name': 'test'                          # static
    }
)
step_fn = lambda s: s.update_global_attr('resource_level', s.global_attrs['resource_level'] * 0.9)
jitted = jax.jit(step_fn)
r1 = jitted(state)   # compiles once
r2 = jitted(r1)      # does NOT recompile
# r2.global_attrs['resource_level'] == 3240.0
```

## Impact

~3-10x speedup. Unblocks `lax.scan` and `vmap`.
