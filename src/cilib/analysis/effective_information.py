"""
Effective Information (EI) and causal-emergence kernel — the reusable measure.

Pure JAX, ``GraphState``-free, ``jit`` / ``vmap``-able. This is the measurement-tier
counterpart to ``engine/environments/commons_metrics.py``: you feed it a transition
operator (a row-stochastic matrix ``T``) and a partition, and it returns scalars. It
imports nothing from ``core`` or any paradigm, so it stays usable as an offline analysis
brick regardless of what produced ``T``.

It is a JAX port of the numpy reference
``…/requisite-variety-emergence/sim/ei_scale_demo.py`` (Hoel 2017 EI; on networks,
Klein & Hoel 2020). The reference's per-row Python loops for coarse-graining are exactly the
linear-algebra object below, for a one-hot (or soft) partition matrix ``S`` (N×K) and a
stationary distribution ``p`` (N,):

    M = D^{-1} · Sᵀ · diag(p) · T · S ,     D = diag(Sᵀ p)               (macro TPM)

with effective information of a row-stochastic ``M`` (uniform intervention, in bits)

    EI(M) = (1/K) Σ_a KL( M[a] ‖ avg )  =  H(avg) − mean_a H(M[a])  =  Det − Deg

    Det = log2 K − mean_a H(M[a])    (how deterministic each cause's effect is)
    Deg = log2 K − H(avg)            (how degenerate / concentrated the average effect is)

Causal emergence is EI(macro) > EI(micro): coarse-graining converts micro degeneracy into
macro determinism. The Kemeny–Snell lumpability ``leak`` (→0 iff strongly lumpable) shares
the same three matmuls, because the block-exit matrix ``X = T·S`` has ``p``-weighted
within-block mean equal to ``M`` itself.

All entropies/logs are base-2 (bits). Functions preserve input dtype — enable
``jax.config.update("jax_enable_x64", True)`` for f64-tight agreement with the numpy reference.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

_TINY = 1e-300  # guards log(0) exactly as the numpy reference (np.maximum(avg, 1e-300))


# ----------------------------------------------------------------------
# Stationary distribution of a row-stochastic operator.
# ----------------------------------------------------------------------

def stationary(T: jax.Array, iters: int = 4000) -> jax.Array:
    """Stationary distribution ``p`` with ``p T = p`` via fixed-iteration power method.

    ``iters`` is static (no data-dependent break) so the function stays ``jit``/``vmap``
    safe; the chains here are small and irreducible (the reference adds uniform noise), so a
    few thousand iterations reach machine precision. Mirrors ``ei_scale_demo.py:stationary``.
    """
    N = T.shape[0]
    p0 = jnp.full((N,), 1.0 / N, dtype=T.dtype)

    def body(_, p):
        p = p @ T
        return p / jnp.sum(p)

    p = jax.lax.fori_loop(0, iters, body, p0)
    return p / jnp.sum(p)


# ----------------------------------------------------------------------
# Partitions.
# ----------------------------------------------------------------------

def partition_to_S(labels: jax.Array, K: Optional[int] = None,
                   dtype=jnp.float32) -> jax.Array:
    """One-hot partition matrix ``S`` (N, K) from integer block ``labels`` (N,).

    ``S[i, k] = 1`` iff node ``i`` is in block ``k``. ``K`` defaults to ``max(labels)+1``
    and must be static under ``jit``.
    """
    labels = jnp.asarray(labels)
    if K is None:
        K = int(jnp.max(labels)) + 1
    return jax.nn.one_hot(labels, K, dtype=dtype)


# ----------------------------------------------------------------------
# Coarse-graining: the one matrix object the rest is built on.
# ----------------------------------------------------------------------

def macro_tpm(T: jax.Array, p: jax.Array, S: jax.Array) -> jax.Array:
    """Stationary-weighted coarse-grained TPM ``M = D^{-1} Sᵀ diag(p) T S`` (K, K).

    Equivalent to the per-row reference ``macro_tpm`` (``ei_scale_demo.py:83``) for any valid
    partition (every block non-empty, ``p>0`` from irreducibility). ``S`` may be soft;
    columns with zero stationary mass yield a zero row (guarded division).
    """
    Pa = S.T @ p                                   # (K,) block stationary mass
    num = S.T @ (p[:, None] * T) @ S               # (K, K)
    Dinv = jnp.where(Pa > 0, 1.0 / jnp.where(Pa > 0, Pa, 1.0), 0.0)
    return Dinv[:, None] * num


def coarse_grain(T: jax.Array, p: jax.Array, labels: jax.Array,
                 K: Optional[int] = None) -> jax.Array:
    """Convenience: ``macro_tpm`` from integer block ``labels`` instead of an ``S`` matrix."""
    return macro_tpm(T, p, partition_to_S(labels, K, dtype=T.dtype))


# ----------------------------------------------------------------------
# Effective information and its Hoel decomposition (bits).
# ----------------------------------------------------------------------

def _row_entropy_bits(M: jax.Array) -> jax.Array:
    """Per-row Shannon entropy in bits, with the ``0·log0 = 0`` convention. Shape (K,)."""
    terms = jnp.where(M > 0, -M * jnp.log2(jnp.where(M > 0, M, 1.0)), 0.0)
    return jnp.sum(terms, axis=1)


def ei_bits(M: jax.Array, intervention: Optional[jax.Array] = None) -> jax.Array:
    """Effective information of a row-stochastic ``M`` in bits, under an explicit intervention
    distribution ``w`` over causes (rows):

        EI = Σ_a w_a · KL( M[a] ‖ M̄_w ),     M̄_w = Σ_a w_a M[a].

    ``intervention=None`` uses the **uniform / maximum-entropy** intervention ``w_a = 1/K`` —
    Hoel's original definition, matching ``ei_scale_demo.py:ei_bits`` exactly (then ``M̄_w`` is the
    plain row-mean and the sum is divided by ``K``). Pass a weight vector (e.g. the chain's macro
    **stationary occupancy** ``Pa = Sᵀp`` normalised, via :func:`macro_stationary`) for the
    *dynamics-derived* intervention the paper claims in §"What is licensed": EI then measures the
    information the system's own typical state carries about its next macro state, not that of a
    stipulated uniform prior. This keeps the intervention distribution (which states we intervene
    on) distinct from the stationary weighting used to *aggregate* the macro TPM in
    :func:`macro_tpm` — the two are different operations.
    """
    K = M.shape[0]
    if intervention is None:
        w = jnp.full((K,), 1.0 / K)                 # weak dtype: promotes to M's dtype in w @ M
    else:
        w = intervention / jnp.sum(intervention)
    avg = w @ M                                     # (K,) intervention-weighted average effect
    avg_safe = jnp.maximum(avg, _TINY)
    kl_rows = jnp.sum(
        jnp.where(M > 0, M * jnp.log2(jnp.where(M > 0, M, 1.0) / avg_safe), 0.0), axis=1)
    return jnp.sum(w * kl_rows)


def macro_stationary(p: jax.Array, S: jax.Array) -> jax.Array:
    """Macro intervention distribution from the micro stationary ``p`` and partition ``S``: the
    normalised block masses ``Pa = Sᵀp``. This is the dynamics-derived weight vector to pass to
    :func:`ei_bits` as ``intervention``."""
    Pa = S.T @ p
    return Pa / jnp.sum(Pa)


def det_bits(M: jax.Array) -> jax.Array:
    """Determinism coefficient ``log2 K − mean_a H(M[a])`` (bits)."""
    K = M.shape[0]
    return jnp.log2(jnp.asarray(K, dtype=M.dtype)) - jnp.mean(_row_entropy_bits(M))


def deg_bits(M: jax.Array) -> jax.Array:
    """Degeneracy coefficient ``log2 K − H(avg)`` (bits). ``EI = Det − Deg``."""
    K = M.shape[0]
    avg = jnp.mean(M, axis=0)
    avg_terms = jnp.where(avg > 0, -avg * jnp.log2(jnp.where(avg > 0, avg, 1.0)), 0.0)
    H_avg = jnp.sum(avg_terms)
    return jnp.log2(jnp.asarray(K, dtype=M.dtype)) - H_avg


def ei_components(M: jax.Array):
    """``(ei, det, deg)`` in bits — exposes the decomposition to diagnose where an EI peak
    comes from (rising determinism vs falling degeneracy across scales)."""
    return ei_bits(M), det_bits(M), deg_bits(M)


# ----------------------------------------------------------------------
# Lumpability (Kemeny–Snell): how genuinely Markov the macro level is.
# ----------------------------------------------------------------------

def leak(T: jax.Array, p: jax.Array, S: jax.Array) -> jax.Array:
    """Stationary-weighted mean std of block-exit probabilities → 0 iff strongly lumpable.

    Strong lumpability requires the block-exit probability ``Σ_{j∈B} T[i,j]`` to be constant
    across ``i ∈ A``. ``X = T S`` is the (N, K) node→block exit matrix; its ``p``-weighted
    within-block mean equals the macro TPM ``M``, so the leak is the within-block weighted
    std of ``X``. Lower = more genuinely Markov at the macro scale. Mirrors
    ``ei_scale_demo.py:leak``.
    """
    Pa = S.T @ p
    Dinv = jnp.where(Pa > 0, 1.0 / jnp.where(Pa > 0, Pa, 1.0), 0.0)
    X = T @ S                                       # (N, K) exit prob to each block
    pX = p[:, None] * X
    M = Dinv[:, None] * (S.T @ pX)                  # (K, K) weighted-mean exit = macro TPM
    E2 = Dinv[:, None] * (S.T @ (p[:, None] * (X * X)))
    var = jnp.maximum(E2 - M * M, 0.0)
    std = jnp.sqrt(var)
    K = S.shape[1]
    return jnp.sum(std) / (K * K)


# ----------------------------------------------------------------------
# Top-level convenience.
# ----------------------------------------------------------------------

def ei_of_partition(T: jax.Array, p: jax.Array, labels: jax.Array,
                    K: Optional[int] = None) -> jax.Array:
    """EI (bits) of the coarse-graining induced by integer block ``labels``."""
    return ei_bits(coarse_grain(T, p, labels, K))
