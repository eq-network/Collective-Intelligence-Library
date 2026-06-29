"""
Initial-state factory for the Commons Harvest substrate.

The spatial grid lives in ``GraphState`` with no core change: the **apple grid is a dynamic global
array** (``global_attrs["apples"]`` (H,W) — a JAX array, so ``GraphState.tree_flatten`` makes it a
traced child that updates across the scan), and **agents are nodes** carrying integer positions in
``node_attrs["pos"]`` (N,2). ``rng_key`` and ``step`` are dynamic global scalars threaded by the
round (the ``_split_key`` pattern).

``make_init_fn`` returns ``key -> GraphState`` and is called *inside* ``run_scan_batch`` under
``jax.vmap``, so it must be vmap/jit-safe: apples are seeded with ``jr.uniform`` over the static
orchard mask, and agent placement uses ``jr.choice(replace=False)`` over the static interior flat
indices — both fixed-shape.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from .config import HarvestConfig, layout


def make_state(cfg: HarvestConfig, key) -> GraphState:
    H, W, N = cfg.height, cfg.width, cfg.n_agents
    wall, orchard, interior_flat = layout(cfg)
    orchard_j = jnp.asarray(orchard)                       # (H,W) bool, constant
    interior_j = jnp.asarray(interior_flat)                # (n_interior,) int, constant

    k_ap, k_pos = jr.split(key)

    # seed apples on a random subset of orchard cells
    draw = jr.uniform(k_ap, (H, W))
    apples = (orchard_j & (draw < cfg.init_apple_frac)).astype(jnp.float32)

    # place agents on distinct interior cells
    chosen = jr.choice(k_pos, interior_j, shape=(N,), replace=False)   # (N,) flat indices
    pos = jnp.stack([chosen // W, chosen % W], axis=1).astype(jnp.int32)  # (N, 2)

    node_types = jnp.zeros(N, dtype=jnp.int32)
    node_attrs = {
        "pos": pos,
        "action": jnp.zeros(N, dtype=jnp.int32),          # last chosen action (for move/zap/trace)
        "frozen": jnp.zeros(N, dtype=jnp.int32),          # steps remaining frozen after a zap
        "last_harvest": jnp.zeros(N, dtype=jnp.float32),
        "cumulative_harvest": jnp.zeros(N, dtype=jnp.float32),
        "rewards": jnp.zeros(N, dtype=jnp.float32),
    }
    global_attrs = {
        "apples": apples,                                  # (H,W) dynamic — the resource
        "rng_key": key,
        "step": jnp.array(0, dtype=jnp.int32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={},                                   # base substrate is networkless
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: HarvestConfig):
    """``key -> GraphState`` for ``run_scan_batch`` (independent per-seed init)."""
    return lambda key: make_state(cfg, key)
