"""
Affiliation-graph primitives — pure single-purpose array ops (jax).

The affiliation graph W (N,N, row-stochastic) is the endogenous institutional structure: row i
is agent i's delegation/trust distribution. ``aggregate`` turns individual votes into per-agent
quotas via W; ``row_softmax`` realizes W from learnable logits.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def row_softmax(logits: jax.Array) -> jax.Array:
    """Row-stochastic affiliation matrix from logits (numerically stable softmax per row)."""
    z = logits - jnp.max(logits, axis=-1, keepdims=True)
    e = jnp.exp(z)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def aggregate(affiliation: jax.Array, values: jax.Array) -> jax.Array:
    """Affiliation-weighted aggregation ``q_i = Σ_j W[i,j] · values_j`` (the institution's
    collective-choice step). W=I → own value (atomized); W=ones/N → global mean (monocentric);
    block-structured W → block mean (polycentric)."""
    return affiliation @ values


def restraint_from_harvest(harvest: jax.Array, max_level: float) -> jax.Array:
    """Restraint signal in [0,1]: 1 = took nothing, 0 = took the max. The slow variable that
    affiliating-with-the-restrained rewards."""
    return jnp.clip(1.0 - harvest / max_level, 0.0, 1.0)
