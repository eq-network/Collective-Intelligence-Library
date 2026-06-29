"""
Tests for the vendored information-form Gaussian / BMR kernel.

These pin the closed-form identities the dynamics rely on (so a refactor of the
transforms can't silently corrupt the math).

Run: python -m pytest engine/paradigms/active_inference/tests/test_primitives.py -q
"""
import jax.numpy as jnp

from cilib.paradigms.active_inference.primitives import (
    info_mean, info_cov, log_evidence, fisher_deposit, fisher_deposit_weighted,
    predictive_logpdf, predictive_logpdf_perchannel, savage_dickey,
    schur_marginalize, zero_edge_prior, softmax, entropy,
)
from cilib.paradigms.active_inference.primitives.gaussian_info import LOG_2PI


def test_log_evidence_known_value():
    # Pi = I, h = 0  ->  logZ = 1/2 [0 - 0 + d log 2pi]
    d = 3
    Pi = jnp.eye(d)
    h = jnp.zeros(d)
    assert jnp.allclose(log_evidence(Pi, h), 0.5 * d * LOG_2PI)


def test_log_evidence_batches():
    Pi = jnp.broadcast_to(jnp.eye(2), (4, 5, 2, 2))
    h = jnp.zeros((4, 5, 2))
    out = log_evidence(Pi, h)
    assert out.shape == (4, 5)
    assert jnp.allclose(out, 0.5 * 2 * LOG_2PI)


def test_info_mean_cov_roundtrip():
    Pi = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    mu = jnp.array([1.0, -2.0])
    h = Pi @ mu
    assert jnp.allclose(info_mean(Pi, h), mu, atol=1e-5)
    assert jnp.allclose(info_cov(Pi) @ Pi, jnp.eye(2), atol=1e-5)


def test_fisher_deposit_identity_operator():
    d = 3
    H = jnp.eye(d)
    o = jnp.array([1.0, 2.0, 3.0])
    J, j = fisher_deposit(H, o, sigma_o=1.0)
    assert jnp.allclose(J, jnp.eye(d))
    assert jnp.allclose(j, o)
    # all-ones weights recover the unweighted deposit
    Jw, jw = fisher_deposit_weighted(H, o, 1.0, jnp.ones(d))
    assert jnp.allclose(Jw, J) and jnp.allclose(jw, j)
    # zero weight on a channel = experiment not run (no info on that coord)
    Jz, _ = fisher_deposit_weighted(H, o, 1.0, jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(Jz, jnp.diag(jnp.array([1.0, 0.0, 0.0])))


def test_predictive_density_peaks_at_match():
    Pi = jnp.eye(2) * 2.0
    mu = jnp.array([1.0, 0.0])
    h = Pi @ mu
    H = jnp.eye(2)
    near = predictive_logpdf(Pi, h, H, jnp.array([1.0, 0.0]), 0.5)
    far = predictive_logpdf(Pi, h, H, jnp.array([-3.0, 0.0]), 0.5)
    assert near > far
    # per-channel: channel 0 (matches) has higher density than channel 1 (off)
    perch = predictive_logpdf_perchannel(Pi, h, H, jnp.array([1.0, 1.0]), 0.5)
    assert perch.shape == (2,)
    assert perch[0] > perch[1]


def test_savage_dickey_no_edit_is_zero():
    # If reduced prior == full prior, the log Bayes factor is exactly 0.
    Pi0 = jnp.array([[2.0, 0.3], [0.3, 2.0]])
    h0 = jnp.array([0.1, -0.2])
    J = jnp.eye(2) * 1.5
    j = jnp.array([0.4, 0.4])
    Pi_post, h_post = Pi0 + J, h0 + j
    dF, Pi_rp, h_rp = savage_dickey(Pi_post, h_post, Pi0, h0, Pi0, h0)
    assert jnp.allclose(dF, 0.0, atol=1e-4)
    # reduced posterior = reduced prior + shared likelihood deposit
    assert jnp.allclose(Pi_rp, Pi0 + J, atol=1e-5)
    assert jnp.allclose(h_rp, h0 + j, atol=1e-5)


def test_savage_dickey_discriminates_edge_support():
    # Prior precision off-diagonal b>0 encodes ANTI-correlation (Σ = Pi^{-1}).
    # Data anti-correlated -> edge supported (lower ΔF, pruning disfavoured);
    # data positively correlated -> edge unsupported (higher ΔF, pruning favoured).
    # The robust, convention-independent claim is the ORDERING.
    Pi0 = jnp.array([[2.0, 1.2], [1.2, 2.0]])
    h0 = jnp.zeros(2)
    J = jnp.eye(2) * 8.0
    Pi0_red = zero_edge_prior(Pi0, 0, 1)

    def dF_for(j):
        Pi_post, h_post = Pi0 + J, h0 + j
        dF, _, _ = savage_dickey(Pi_post, h_post, Pi0, h0, Pi0_red, h0)
        return dF

    dF_consistent = dF_for(jnp.array([4.0, -4.0]))   # anti-correlated (edge expects this)
    dF_inconsistent = dF_for(jnp.array([4.0, 4.0]))  # positively correlated
    assert dF_inconsistent > dF_consistent, (
        f"BMR failed to discriminate: incons={float(dF_inconsistent)} "
        f"cons={float(dF_consistent)}")


def test_schur_marginalize_independent_no_fillin():
    # Diagonal (independent) precision: marginalizing one coord leaves the other
    # untouched (no carry-over fill-in).
    Pi = jnp.diag(jnp.array([2.0, 3.0]))
    h = jnp.array([4.0, 9.0])
    keep = jnp.array([0]); drop = jnp.array([1])
    Pi_m, h_m = schur_marginalize(Pi, h, keep, drop)
    assert jnp.allclose(Pi_m, jnp.array([[2.0]]))
    assert jnp.allclose(h_m, jnp.array([4.0]))


def test_softmax_entropy():
    assert jnp.allclose(softmax(jnp.zeros(4)), jnp.full(4, 0.25))
    assert jnp.allclose(entropy(jnp.zeros(4)), jnp.log(4.0), atol=1e-5)       # uniform = max
    assert entropy(jnp.array([20.0, 0.0])) < 1e-3                             # peaked = ~0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all primitives tests passed")
