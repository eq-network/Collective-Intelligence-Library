"""
Tests for the JAX Effective-Information kernel.

Strategy: pin the JAX matrix-form kernel to a faithful **numpy reimplementation of the paper
reference** (``…/requisite-variety-emergence/sim/ei_scale_demo.py``). The reference's
group-list / per-row loops and the kernel's ``M = D^{-1} Sᵀ diag(p) T S`` form must agree to
f64 tolerance on (a) random operators, (b) random partitions, and (c) the actual paper
network — and the reference's three qualitative results must reproduce. This guarantees a
later refactor of the kernel can't silently corrupt the measure.

Run: python -m pytest engine/analysis/tests/test_effective_information.py -q
"""
import jax.numpy as jnp
import numpy as np

from cilib.analysis.effective_information import (
    stationary, partition_to_S, macro_tpm, coarse_grain,
    ei_bits, det_bits, deg_bits, leak,
)

# NOTE: we deliberately do NOT enable jax_enable_x64 here — doing so at import time pollutes
# the whole pytest session (other suites assume float32). The kernel runs in float32; we test
# JAX/numpy *equivalence* to 1e-4, which is far inside the float32 noise floor for EI in bits.
# The offline pipeline (causal_emergence.py) carries its own numpy-float64 path for exact
# reference reproduction.
ATOL = 1e-4


# ======================================================================
# Faithful numpy reference (verbatim semantics from ei_scale_demo.py)
# ======================================================================

NB, BS = 4, 4
N = NB * BS
NOISE = 0.05


def _block_of(i):
    return i // BS


def ref_build_W(c=0.0, seed=7):
    rng = np.random.default_rng(seed)
    inb, cyc, eps = 0.55, 1.00, 0.01
    W = np.full((N, N), eps)
    for i in range(N):
        bi = _block_of(i)
        succ = (bi + 1) % NB
        for j in range(bi * BS, bi * BS + BS):
            if j != i:
                W[i, j] += inb / (BS - 1)
        for k in range(BS):
            W[i, succ * BS + k] += cyc * (1 - c) / BS
        if c > 0 and bi != 0:
            for k in range(BS):
                W[i, k] += cyc * c / BS
    W *= (0.92 + 0.16 * rng.random((N, N)))
    return W


def _row_norm(W):
    r = W.sum(axis=1, keepdims=True)
    r[r == 0] = 1.0
    return W / r


def ref_build_T(c=0.0, seed=7):
    return (1.0 - NOISE) * _row_norm(ref_build_W(c, seed)) + NOISE / N


def ref_stationary(T, iters=20000, tol=1e-14):
    n = T.shape[0]
    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        pn = p @ T
        if np.abs(pn - p).sum() < tol:
            break
        p = pn
    return p / p.sum()


def ref_macro_tpm(T, p, groups):
    n = len(groups)
    M = np.zeros((n, n))
    for a, ga in enumerate(groups):
        Pa = p[ga].sum()
        wa = (p[ga] / Pa) if Pa > 0 else np.full(len(ga), 1.0 / len(ga))
        out = (wa[:, None] * T[ga, :]).sum(axis=0)
        for b, gb in enumerate(groups):
            M[a, b] = out[gb].sum()
    return M


def ref_ei_bits(M):
    avg = M.mean(axis=0)
    tot = 0.0
    for i in range(M.shape[0]):
        pi = M[i]
        m = pi > 0
        tot += np.sum(pi[m] * np.log2(pi[m] / np.maximum(avg[m], 1e-300)))
    return tot / M.shape[0]


def ref_leak(T, p, groups):
    n = T.shape[0]
    tot = 0.0
    for A in groups:
        for B in groups:
            exit_probs = T[np.ix_(A, range(n))][:, B].sum(axis=1)
            w = p[A] / p[A].sum() if p[A].sum() > 0 else np.full(len(A), 1 / len(A))
            mean = (w * exit_probs).sum()
            tot += np.sqrt((w * (exit_probs - mean) ** 2).sum())
    return tot / (len(groups) ** 2)


def ref_true_blocks():
    return [list(range(b * BS, b * BS + BS)) for b in range(NB)]


def _groups_to_labels(groups, n):
    labels = np.empty(n, dtype=np.int64)
    for k, g in enumerate(groups):
        for i in g:
            labels[i] = k
    return labels


def _labels_to_groups(labels, K):
    return [list(np.where(labels == k)[0]) for k in range(K)]


def _random_T(n, seed):
    rng = np.random.default_rng(seed)
    R = rng.random((n, n)) + 1e-3
    return R / R.sum(axis=1, keepdims=True)


def _random_labels(n, K, seed):
    """Random K-partition with every block guaranteed non-empty."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, size=n)
    labels[:K] = np.arange(K)          # ensure each block occupied
    rng.shuffle(labels)
    return labels


def _S(labels, K):
    return partition_to_S(jnp.asarray(labels), K, dtype=jnp.float32)


# ======================================================================
# (i) JAX matrix-form == numpy loop-form  (random operators / partitions)
# ======================================================================

def test_stationary_matches_reference():
    T = ref_build_T(0.0)
    p_jax = np.asarray(stationary(jnp.asarray(T)))
    p_ref = ref_stationary(T)
    assert np.allclose(p_jax, p_ref, atol=ATOL)
    # fixed point: p T == p
    assert np.allclose(p_jax @ T, p_jax, atol=ATOL)


def test_macro_tpm_matches_reference_random():
    for seed in (0, 1, 2):
        n = 12
        T = _random_T(n, seed)
        p = ref_stationary(T)
        for K in (2, 3, 4):
            labels = _random_labels(n, K, seed * 10 + K)
            groups = _labels_to_groups(labels, K)
            M_ref = ref_macro_tpm(T, p, groups)
            M_jax = np.asarray(macro_tpm(jnp.asarray(T), jnp.asarray(p), _S(labels, K)))
            assert np.allclose(M_jax, M_ref, atol=ATOL), (seed, K)
            # macro rows are proper distributions
            assert np.allclose(M_jax.sum(axis=1), 1.0, atol=ATOL)


def test_ei_and_leak_match_reference_random():
    for seed in (3, 4, 5):
        n = 16
        T = _random_T(n, seed)
        p = ref_stationary(T)
        for K in (2, 4, 8):
            labels = _random_labels(n, K, seed * 7 + K)
            groups = _labels_to_groups(labels, K)
            M_ref = ref_macro_tpm(T, p, groups)
            S = _S(labels, K)
            ei_jax = float(ei_bits(macro_tpm(jnp.asarray(T), jnp.asarray(p), S)))
            leak_jax = float(leak(jnp.asarray(T), jnp.asarray(p), S))
            assert np.isclose(ei_jax, ref_ei_bits(M_ref), atol=ATOL), (seed, K)
            assert np.isclose(leak_jax, ref_leak(T, p, groups), atol=ATOL), (seed, K)


# ======================================================================
# (iii) Hoel decomposition and known-value identities
# ======================================================================

def test_ei_equals_det_minus_deg():
    rng = np.random.default_rng(11)
    for _ in range(20):
        K = int(rng.integers(2, 9))
        R = rng.random((K, K)) + 1e-3
        M = jnp.asarray(R / R.sum(axis=1, keepdims=True))
        ei = float(ei_bits(M))
        dd = float(det_bits(M) - deg_bits(M))
        assert np.isclose(ei, dd, atol=1e-5)


def test_singleton_partition_recovers_operator():
    T = ref_build_T(0.0)
    p = ref_stationary(T)
    labels = np.arange(N)
    M = np.asarray(coarse_grain(jnp.asarray(T), jnp.asarray(p), jnp.asarray(labels), N))
    assert np.allclose(M, T, atol=ATOL)


def test_trivial_one_block_is_zero_ei():
    T = ref_build_T(0.0)
    p = ref_stationary(T)
    S = _S(np.zeros(N, dtype=np.int64), 1)
    M = macro_tpm(jnp.asarray(T), jnp.asarray(p), S)
    assert M.shape == (1, 1)
    assert np.isclose(float(ei_bits(M)), 0.0, atol=ATOL)
    assert np.isclose(float(leak(jnp.asarray(T), jnp.asarray(p), S)), 0.0, atol=ATOL)


def test_uniform_operator_has_zero_ei():
    # All rows identical -> every row equals the average -> no effective information.
    T = jnp.full((N, N), 1.0 / N, dtype=jnp.float32)
    p = stationary(T)
    for K in (2, 4):
        labels = _random_labels(N, K, K)
        ei = float(ei_bits(macro_tpm(T, p, _S(labels, K))))
        assert np.isclose(ei, 0.0, atol=ATOL)


# ======================================================================
# Reference network: pin to the paper object + reproduce results (A) and (B)
# ======================================================================

def test_reference_network_pins_and_result_A():
    """Institutional partition is causally privileged AND most lumpable vs random (result A),
    and the JAX value equals the numpy reference on the actual paper network."""
    T = ref_build_T(0.0)
    p = ref_stationary(T)
    true_groups = ref_true_blocks()
    true_labels = _groups_to_labels(true_groups, N)

    S_true = _S(true_labels, NB)
    ei_true = float(ei_bits(macro_tpm(jnp.asarray(T), jnp.asarray(p), S_true)))
    leak_true = float(leak(jnp.asarray(T), jnp.asarray(p), S_true))

    # (pin) JAX == numpy reference on the paper network
    assert np.isclose(ei_true, ref_ei_bits(ref_macro_tpm(T, p, true_groups)), atol=ATOL)
    assert np.isclose(leak_true, ref_leak(T, p, true_groups), atol=ATOL)

    # (result A) institutional partition beats random partitions of the same size K=4
    rng = np.random.default_rng(123)
    rand_ei, rand_leak = [], []
    for _ in range(200):
        perm = rng.permutation(N)
        labels = np.empty(N, dtype=np.int64)
        for g in range(NB):
            labels[perm[g * BS:(g + 1) * BS]] = g
        S = _S(labels, NB)
        rand_ei.append(float(ei_bits(macro_tpm(jnp.asarray(T), jnp.asarray(p), S))))
        rand_leak.append(float(leak(jnp.asarray(T), jnp.asarray(p), S)))
    assert ei_true > np.mean(rand_ei)       # causally privileged
    assert leak_true < np.mean(rand_leak)   # most Markov / lumpable


def test_reference_network_shelf_then_cliff_result_B():
    """EI is retained at the meso (institutional) scale and collapses to 0 at the macro
    (single-block) scale — the 'viable shelf then cliff' (result B)."""
    T = ref_build_T(0.0)
    p = ref_stationary(T)
    Tj, pj = jnp.asarray(T), jnp.asarray(p)

    ei_meso = float(ei_bits(macro_tpm(Tj, pj, _S(_groups_to_labels(ref_true_blocks(), N), NB))))
    ei_macro = float(ei_bits(macro_tpm(Tj, pj, _S(np.zeros(N, dtype=np.int64), 1))))

    assert ei_macro < 1e-9          # full aggregation destroys all EI (the cliff)
    assert ei_meso > 0.1            # the institutional scale retains real EI (the shelf)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all effective_information tests passed")
