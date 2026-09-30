"""
Metrics for the IO Economy — the demand-attribution decomposition and its companions.

The headline is the standard IO attribution (Miller & Blair): activity ultimately
driven by human final demand is ``x_H = (I−A)⁻¹ d_H``, and the **human demand share**
is ``1ᵀx_H / 1ᵀ(I−A)⁻¹(d_H + d_AI)`` — "what fraction of the economy still runs for
people". Because the substrate's fixed point IS the Leontief solution, this metric is
exact accounting, not an estimate; hypothetical extraction (drop ``d_H``, read the
output loss) is the same algebra, which is what makes the counterfactual instrument
analytically checkable here (the register's rung for ``environments/counterfactual.py``).

``spectral_margin`` = 1 − ρ(A) is the distance to the reproduction boundary of the
intermediate loop — under labor→cognition substitution at cost parity, column sums
climb toward 1, so the margin closes BY the accounting identity (the diagnostics
thread's operator, here as the model itself rather than a linearization). Power
iteration, not ``eigvals``: A is nonnegative, so the Perron root is the limit of the
iteration norm, and it stays backend-agnostic under ``vmap``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from cilib.metrics.identity import kind


def attribution_series(trace):
    """(T,) human demand share per tick: 1ᵀ(I−A)⁻¹d_H over 1ᵀ(I−A)⁻¹(d_H+d_AI)."""
    A = trace["technical"]                                          # (T, N, N)
    eye = jnp.eye(A.shape[-1])
    d_h = trace["demand_h"][..., None]                              # (T, N, 1)
    d_all = d_h + trace["demand_ai"][..., None]
    x_h = jnp.linalg.solve(eye - A, d_h)[..., 0]
    x_all = jnp.linalg.solve(eye - A, d_all)[..., 0]
    return jnp.sum(x_h, axis=-1) / jnp.maximum(jnp.sum(x_all, axis=-1), 1e-8)


def spectral_radius(A, iters: int = 60):
    """Perron root of a nonnegative matrix by power iteration (vmap/backend safe)."""
    n = A.shape[-1]

    def body(v, _):
        w = A @ v
        norm = jnp.linalg.norm(w) + 1e-12
        return w / norm, norm

    _, norms = jax.lax.scan(body, jnp.full((n,), 1.0 / n), None, length=iters)
    return norms[-1]


def _late(series):
    return series[3 * series.shape[0] // 4:]


def make_metrics(cfg):
    """Trace -> scalar reductions (harness vmaps these over the seed axis)."""

    @kind("share")
    def human_demand_share(trace):
        return jnp.mean(_late(attribution_series(trace)))

    @kind("level")
    def wage_bill_late(trace):
        return jnp.mean(_late(trace["wage_bill"]))

    @kind("level")
    def unsupervised_ai_spending(trace):
        """Mean late per-tick resources allocated by the AI loop's own rule — demand
        that passes through no human decision (Gradual Disempowerment §6.2.1's 'scale
        of unsupervised AI spending', computable here by construction)."""
        return jnp.mean(_late(jnp.sum(trace["demand_ai"], axis=-1)))

    @kind("index")
    def spectral_margin(trace):
        return 1.0 - spectral_radius(trace["technical"][-1])

    return {
        "human_demand_share": human_demand_share,
        "wage_bill_late": wage_bill_late,
        "unsupervised_ai_spending": unsupervised_ai_spending,
        "spectral_margin": spectral_margin,
    }
