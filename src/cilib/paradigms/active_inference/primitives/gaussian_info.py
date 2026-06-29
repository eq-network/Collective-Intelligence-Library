"""
Information-form Gaussian linear algebra — the reusable array kernel.

VENDORED, essentially verbatim, from
``C:/GitHub/Paradigm_Shift_Act_Inf/src/structural/linalg.py`` (the "paradigm
shift" / changing-networked-mind model), with the named-basis ``GaussianBeliefNet``
wrapper deliberately dropped: in this library belief state lives in
``GraphState.node_attrs`` as raw arrays, so these stay pure functions on bare
arrays. Batching over agents/candidates is done by the caller via ``jax.vmap``
(see ``transforms.py``), so every function here is written for a single Gaussian.

Information form. A Gaussian over ``x in R^d`` is carried as ``(Pi, h)`` with

    p(x) ∝ exp( -1/2 x^T Pi x + h^T x ),     Pi = Σ^{-1},  h = Pi μ.

The payoff of these coordinates: *learning is addition* (``Pi += J``), *fusion is
addition* (``Pi1 + Pi2``), *marginalizing is a Schur complement*, and the
log-normalizer is closed form. Each function below is one of those identities.

Why vendor rather than import: the source repo is not an installed package, and
this kernel is the one piece we want lifted exactly (re-deriving Savage–Dickey /
Schur identities is pure downside risk). It is ~200 lines of pure JAX with zero
scenario coupling.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

LOG_2PI = float(jnp.log(2.0 * jnp.pi))
SPIKE_PRECISION = 1e6  # precision used to pin a coordinate to a sharp value (BMR)


# ----------------------------------------------------------------------
# Solves (mean / covariance from information form).
# ----------------------------------------------------------------------

def info_mean(Pi: jax.Array, h: jax.Array) -> jax.Array:
    """Posterior mean ``μ = Pi^{-1} h``, solved (never inverted). Batches over any
    leading axes (``Pi`` ``(..., d, d)``, ``h`` ``(..., d)``)."""
    return jnp.linalg.solve(Pi, h[..., None]).squeeze(-1)


def info_cov(Pi: jax.Array) -> jax.Array:
    """Posterior covariance ``Σ = Pi^{-1}`` — the one function that forms the
    inverse explicitly. Prefer ``info_mean`` / ``log_evidence`` (which solve)."""
    return jnp.linalg.inv(Pi)


# ----------------------------------------------------------------------
# The closed-form log-normalizer.
# ----------------------------------------------------------------------

def log_evidence(Pi: jax.Array, h: jax.Array) -> jax.Array:
    """Closed-form Gaussian log-normalizer

        logZ = 1/2 [ h^T Pi^{-1} h  -  logdet(Pi)  +  d log(2π) ].

    Uses ``slogdet`` (stable) and ``solve`` (no explicit inverse). Batches over
    any leading axes of ``Pi`` (``(..., d, d)``) and ``h`` (``(..., d)``)."""
    _sign, logabsdet = jnp.linalg.slogdet(Pi)
    sol = jnp.linalg.solve(Pi, h[..., None]).squeeze(-1)
    quad = jnp.einsum('...i,...i->...', h, sol)
    d = Pi.shape[-1]
    return 0.5 * (quad - logabsdet + d * LOG_2PI)


# ----------------------------------------------------------------------
# Learn / fuse — both are addition in information form.
# ----------------------------------------------------------------------

def add_information(Pi, h, J, j):
    """Deposit information (LEARN) or fuse an independent belief (FUSE): the same
    running sum ``Pi + J``, ``h + j``."""
    return Pi + J, h + j


def row_stochastic(weights: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Normalise each row to sum to 1 — closed-neighbourhood fusion weights."""
    return weights / (weights.sum(axis=-1, keepdims=True) + eps)


# ----------------------------------------------------------------------
# Likelihood: the Fisher-information deposit of one linear-Gaussian observation.
# ----------------------------------------------------------------------

def fisher_deposit(H: jax.Array, o: jax.Array, sigma_o: float):
    """Information from one observation ``o = H x + ε``, ``ε ~ N(0, σ_o^2 I)``:

        J = H^T H / σ_o^2   (d, d),    j = H^T o / σ_o^2   (d,).

    ``H`` is (m, d), ``o`` is (m,)."""
    inv_var = 1.0 / (sigma_o ** 2)
    return (H.T @ H) * inv_var, (H.T @ o) * inv_var


def fisher_deposit_weighted(H: jax.Array, o: jax.Array, sigma_o: float,
                            weights: jax.Array):
    """Per-channel attention-weighted deposit: ``J = H^T diag(w) H / σ_o^2``,
    ``j = H^T diag(w) o / σ_o^2``. ``weights`` (m,) in [0, ∞): ``w_k`` is the gain
    on channel ``k`` — ``1`` full, ``0`` the experiment is not run. All-ones
    recovers ``fisher_deposit``. Each row scaled by ``sqrt(w_k)``."""
    sw = jnp.sqrt(jnp.clip(weights, 0.0, None))
    return fisher_deposit(sw[:, None] * H, sw * o, sigma_o)


def predictive_logpdf(Pi: jax.Array, h: jax.Array, H: jax.Array,
                      o: jax.Array, sigma_o: float) -> jax.Array:
    """One-step-ahead (prequential) predictive log-density ``log p(o | belief)``.

    Under belief ``(Pi, h)`` the predictive for ``o = H x + ε`` is Gaussian with
    mean ``H μ`` and covariance ``H Σ H^T + σ_o^2 I`` (``μ = Pi^{-1}h``,
    ``Σ = Pi^{-1}``). This is the quantity that, accumulated, separates rival
    structures: a wiring that predicts the data better gains log-weight. ``H`` is
    (m, d), ``o`` is (m,)."""
    mu = info_mean(Pi, h)
    Sigma = info_cov(Pi)
    pred_mean = H @ mu
    m = H.shape[0]
    pred_cov = H @ Sigma @ H.T + (sigma_o ** 2) * jnp.eye(m)
    resid = o - pred_mean
    _sign, logdet = jnp.linalg.slogdet(pred_cov)
    quad = resid @ jnp.linalg.solve(pred_cov, resid)
    return -0.5 * (quad + logdet + m * LOG_2PI)


def predictive_logpdf_perchannel(Pi: jax.Array, h: jax.Array, H: jax.Array,
                                 o: jax.Array, sigma_o: float) -> jax.Array:
    """Per-channel MARGINAL predictive log-densities ``log p(o_k | belief)`` — one
    per observation channel, shape (m,).

    This is the theory-laden scoring quantity: weighting these by an agent's
    per-channel attention ``w_obs`` and summing gives an attention-weighted
    prequential score, so two camps reading the same world through different
    channels can accumulate evidence for different wirings. (Uses the marginal
    per-channel predictive variance ``diag(H Σ H^T) + σ_o^2``, ignoring cross-channel
    covariance — the right granularity for per-channel attention.)"""
    mu = info_mean(Pi, h)
    Sigma = info_cov(Pi)
    pred_mean = H @ mu                                      # (m,)
    pred_var = jnp.sum((H @ Sigma) * H, axis=1) + sigma_o ** 2   # (m,)
    return -0.5 * ((o - pred_mean) ** 2 / pred_var + jnp.log(pred_var) + LOG_2PI)


# ----------------------------------------------------------------------
# Reduction: Schur complement (marginalize), and prior edits for BMR.
# ----------------------------------------------------------------------

def schur_marginalize(Pi: jax.Array, h: jax.Array,
                      keep: jax.Array, drop: jax.Array):
    """REDUCE by marginalizing out the ``drop`` block, exactly:

        Pi_a^marg = Pi_aa - Pi_ab Pi_bb^{-1} Pi_ba
        h_a^marg  = h_a   - Pi_ab Pi_bb^{-1} h_b.

    ``keep`` / ``drop`` integer index arrays. Survivors keep the Schur complement;
    the subtracted term is the carry-over fill-in among the dropped block's
    neighbours (the "ghost" of a removed hub)."""
    Pi_aa = Pi[jnp.ix_(keep, keep)]
    Pi_ab = Pi[jnp.ix_(keep, drop)]
    Pi_bb = Pi[jnp.ix_(drop, drop)]
    Pi_ba = Pi[jnp.ix_(drop, keep)]
    Pi_marg = Pi_aa - Pi_ab @ jnp.linalg.solve(Pi_bb, Pi_ba)
    h_marg = h[keep] - Pi_ab @ jnp.linalg.solve(Pi_bb, h[drop])
    return Pi_marg, h_marg


# ----------------------------------------------------------------------
# Bayesian Model Reduction: closed-form log Bayes factor for a prior edit.
# ----------------------------------------------------------------------

def savage_dickey(Pi_post, h_post, Pi0, h0, Pi0_red, h0_red):
    """REDUCE by pruning: closed-form log Bayes factor ΔF for swapping the prior
    ``(Pi0, h0)`` for a sharper ``(Pi0_red, h0_red)`` under the SAME likelihood
    (matrix Savage–Dickey / Friston–Penny post-hoc identity).

    Shared likelihood deposit ``J = Pi_post - Pi0``, ``j = h_post - h0`` ⇒ reduced
    posterior ``(Pi0_red + J, h0_red + j)`` and

        ΔF = [logZ(red_post) - logZ(red_prior)] - [logZ(post) - logZ(prior)].

    Exact for Gaussians. ``ΔF > 0`` ⇒ the data are content with the edit (reduced
    model favoured); ``< 0`` ⇒ the data hold the full model. Returns
    ``(delta_F, Pi_red_post, h_red_post)``."""
    J = Pi_post - Pi0
    j = h_post - h0
    Pi_rp = Pi0_red + J
    h_rp = h0_red + j
    delta_F = (
        (log_evidence(Pi_rp, h_rp) - log_evidence(Pi0_red, h0_red))
        - (log_evidence(Pi_post, h_post) - log_evidence(Pi0, h0))
    )
    return delta_F, Pi_rp, h_rp


def spike_prior(Pi: jax.Array, h: jax.Array, idx: jax.Array,
                value: float = 0.0, spike: float = SPIKE_PRECISION):
    """Build a reduced prior pinning coordinates ``idx`` to a sharp ``value``:
    zero their couplings (row/col), set diagonal to ``spike`` with potential
    ``spike * value``. ``value = 0`` prunes toward absence."""
    Pi = Pi.at[idx, :].set(0.0)
    Pi = Pi.at[:, idx].set(0.0)
    Pi = Pi.at[idx, idx].set(spike)
    h = h.at[idx].set(spike * value)
    return Pi, h


def zero_edge_prior(Pi: jax.Array, u: int, v: int) -> jax.Array:
    """Pin a single coupling ``(u, v)`` to zero (the edge vanishes), leaving
    everything else intact — a rank-2 symmetric edit."""
    return Pi.at[u, v].set(0.0).at[v, u].set(0.0)


# ----------------------------------------------------------------------
# Posterior-over-structures helpers (the represented-rivals layer).
# ----------------------------------------------------------------------

def softmax(logw: jax.Array, axis: int = -1) -> jax.Array:
    """Numerically stable softmax of log-weights -> q(m), the posterior over
    candidate structures."""
    z = logw - jnp.max(logw, axis=axis, keepdims=True)
    e = jnp.exp(z)
    return e / jnp.sum(e, axis=axis, keepdims=True)


def entropy(logw: jax.Array, axis: int = -1) -> jax.Array:
    """Shannon entropy (nats) of ``q(m) = softmax(logw)`` — "crisis as a torn
    state" is a peak in this quantity."""
    q = softmax(logw, axis=axis)
    return -jnp.sum(q * jnp.log(q + 1e-30), axis=axis)
