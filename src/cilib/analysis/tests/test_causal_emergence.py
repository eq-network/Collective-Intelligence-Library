"""
Tests for the offline causal-emergence pipeline.

Two roles:
- Phase 3 (reproduction): pin the numpy pipeline to the paper's published reference numbers
  (sim/results.txt) and to the JAX kernel, and check emergent-partition / Track-B recovery.
- Phase 2 (estimator gate): the TPM-from-trajectory estimator must recover the true EI
  (consistency) and must NOT manufacture a false interior EI peak on an i.i.d. null at the
  sampling regime used — the finite-sample-bias artifact that would fake causal emergence.

Run: python -m pytest engine/analysis/tests/test_causal_emergence.py -q
"""
import numpy as np
import jax.numpy as jnp

from cilib.analysis import causal_emergence as ce
from cilib.analysis import effective_information as ei_jax


# =============================================================================
# Phase 3 — reproduce the published reference numbers (sim/results.txt)
# =============================================================================

def test_reference_A_privileged_partition():
    """(A) institutional partition: EI=0.6804, leak=0.0038, both at the 100th percentile."""
    T = ce.build_reference_T(0.0)
    p = ce.stationary(T)
    meso = np.repeat(np.arange(4), 4)            # the institutional partition
    res = ce.null_compare(T, p, meso, n_null=500, seed=123)
    assert abs(res["ei"] - 0.6804) < 2e-3, res["ei"]
    assert abs(res["leak"] - 0.0038) < 2e-3, res["leak"]
    assert res["ei_percentile"] == 100.0
    assert res["leak_percentile"] == 100.0
    assert abs(res["rand_ei_mean"] - 0.0343) < 1e-2, res["rand_ei_mean"]


def test_reference_B_shelf_then_cliff():
    """(B) EI retained micro->meso, then collapses: 0.7794/0.7035/0.6804/0.0700/0.0000."""
    T = ce.build_reference_T(0.0)
    p = ce.stationary(T)
    hier = ce.reference_scale_hierarchy(16, 4)
    curve = ce.ei_curve(T, p, hier)
    expected = [0.7794, 0.7035, 0.6804, 0.0700, 0.0000]
    for got, exp, (lbl, _) in zip(curve, expected, hier):
        assert abs(got - exp) < 2e-3, (lbl, got, exp)


def test_reference_C_capture_qualitative():
    """(C) rising capture collapses institutional EI and concentrates Shapley-EI.

    Fresh-RNG-per-c gives a different absolute curve than the paper's shared-RNG sweep, so we
    assert the robust qualitative signature, not the exact sweep numbers.
    """
    meso = np.repeat(np.arange(4), 4)
    eis, shares, totals = [], [], []
    for c in (0.0, 0.3, 0.6, 0.9):
        T = ce.build_reference_T(c)
        p = ce.stationary(T)
        eis.append(ce.ei_of_labels(T, p, meso, 4))
        phi = ce.shapley_ei(T, p, meso)
        shares.append(ce.max_phi_share(phi))
        totals.append(float(phi.sum()))
    # institutional EI falls monotonically as capture rises
    assert eis[0] > 0.6 and eis[-1] < 0.45
    assert all(eis[i] >= eis[i + 1] - 1e-6 for i in range(len(eis) - 1)), eis
    # healthy Shapley is near-even (each ~1/4), capture concentrates it
    phi0 = ce.shapley_ei(ce.build_reference_T(0.0), ce.stationary(ce.build_reference_T(0.0)), meso)
    assert phi0.min() > 0.14 and phi0.max() < 0.20, phi0
    assert shares[-1] > shares[0]                 # capture concentrates the attribution
    assert totals[-1] < totals[0]                 # total causal power falls


def test_numpy_jax_kernel_agreement():
    """The numpy pipeline and the float32 JAX kernel agree to 1e-3 on the reference network."""
    T = ce.build_reference_T(0.0)
    p = ce.stationary(T)
    meso = np.repeat(np.arange(4), 4)
    S = ce.partition_to_S(meso, 4)
    M_np = ce.macro_tpm(T, p, S)
    # numpy EI/leak vs JAX EI/leak on the same coarse-graining
    ei_np = ce.ei_bits(M_np)
    ei_j = float(ei_jax.ei_bits(jnp.asarray(M_np)))
    assert abs(ei_np - ei_j) < 1e-3
    leak_np = ce.leak(T, p, S)
    leak_j = float(ei_jax.leak(jnp.asarray(T), jnp.asarray(p), jnp.asarray(S)))
    assert abs(leak_np - leak_j) < 1e-3
    # EI = det - deg
    det, deg = ce.det_deg_bits(M_np)
    assert abs(ei_np - (det - deg)) < 1e-9


def test_emergent_partition_recovers_clean_blocks():
    """Greedy modularity recovers a clean planted block structure."""
    n_blocks, bs = 4, 4
    N = n_blocks * bs
    W = np.full((N, N), 0.02)
    for b in range(n_blocks):
        W[b * bs:(b + 1) * bs, b * bs:(b + 1) * bs] = 1.0
    labels = ce.emergent_partition(W)
    assert len(np.unique(labels)) == n_blocks
    # each planted block is internally homogeneous in the recovered labelling
    for b in range(n_blocks):
        block = labels[b * bs:(b + 1) * bs]
        assert len(np.unique(block)) == 1, (b, block)


def test_trackB_recovers_blocks_from_trajectories():
    """Track-B coupling estimated from block-correlated trajectories recovers the blocks."""
    rng = np.random.default_rng(0)
    n_blocks, bs, Tn = 4, 4, 4000
    N = n_blocks * bs
    n_states = 3
    bins = np.zeros((Tn, N), dtype=int)
    for t in range(Tn):
        for b in range(n_blocks):
            latent = rng.integers(0, n_states)              # block's shared latent state
            for a in range(b * bs, (b + 1) * bs):
                bins[t, a] = latent if rng.random() < 0.85 else rng.integers(0, n_states)
    T_behav = ce.coupling_transition_from_trajectories(bins, n_states=n_states)
    labels = ce.emergent_partition(T_behav)
    assert len(np.unique(labels)) == n_blocks
    for b in range(n_blocks):
        block = labels[b * bs:(b + 1) * bs]
        assert len(np.unique(block)) == 1, (b, block)


# =============================================================================
# Phase 2 — the estimator gate
# =============================================================================

def test_estimator_recovers_true_ei():
    """As trajectory length grows, EI of the estimated TPM converges to the true EI."""
    rng = np.random.default_rng(1)
    S = 5
    R = rng.random((S, S)) + 0.05
    T_true = R / R.sum(axis=1, keepdims=True)
    p = ce.stationary(T_true)
    ei_true = ce.ei_bits(T_true)                            # micro EI (singletons => M = T)
    # sample a long trajectory from T_true
    n = 200_000
    traj = np.zeros(n, dtype=int)
    s = 0
    u = rng.random(n)
    cdf = np.cumsum(T_true, axis=1)
    for t in range(1, n):
        traj[t] = np.searchsorted(cdf[traj[t - 1]], u[t])
    T_hat = ce.estimate_tpm(traj, S, laplace=1.0)
    ei_hat = ce.ei_bits(T_hat)
    assert abs(ei_hat - ei_true) < 0.05, (ei_hat, ei_true)


def test_no_false_emergence_on_iid_null():
    """GATE: an i.i.d. process has ~zero EI at the micro scale AND no interior peak across
    coarse-grainings, at the sampling regime we use. If the estimator manufactured determinism
    from undersampling, EI would appear > 0 and a spurious meso peak could form."""
    rng = np.random.default_rng(2)
    S = 8                                                   # state space
    n = 40_000                                              # adequate sampling (~5000/state)
    traj = rng.integers(0, S, size=n)                       # i.i.d. uniform: no structure
    T_hat = ce.estimate_tpm(traj, S, laplace=1.0)
    p = ce.stationary(T_hat)
    # micro EI ~ 0 (rows ~ uniform ~ average)
    assert ce.ei_bits(T_hat) < 0.03, ce.ei_bits(T_hat)
    # coarse-grain the state space at several granularities -> EI stays ~0, no interior peak
    hier = [("micro", np.arange(S)),
            ("4", np.repeat(np.arange(4), S // 4)),
            ("2", (np.arange(S) >= S // 2).astype(int)),
            ("1", np.zeros(S, dtype=int))]
    curve = ce.ei_curve(T_hat, p, hier)
    diag = ce.interior_max(curve)
    assert max(curve) < 0.03, curve                         # no false causal emergence anywhere
    assert diag["prominence"] < 0.02, diag                  # no spurious interior peak


def test_undersampling_bias_is_real_but_controlled():
    """Positive control: severe undersampling inflates apparent EI (the artifact exists), while
    adequate sampling suppresses it. Documents the bias the gate above guards against."""
    rng = np.random.default_rng(3)
    S = 8
    # severe undersampling: far fewer transitions than S^2
    short = rng.integers(0, S, size=40)
    ei_short = ce.ei_bits(ce.estimate_tpm(short, S, laplace=1.0))
    # adequate sampling
    long = rng.integers(0, S, size=40_000)
    ei_long = ce.ei_bits(ce.estimate_tpm(long, S, laplace=1.0))
    assert ei_short > ei_long                               # the bias is real...
    assert ei_long < 0.03                                   # ...and vanishes with enough data


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all causal_emergence tests passed")
