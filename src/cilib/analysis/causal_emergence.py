"""
Offline causal-emergence pipeline — the analysis tier (numpy, float64).

This is the offline counterpart to the JAX kernel in ``effective_information.py``. It is pure
numpy so it can reproduce the paper-reference numbers to full precision (the JAX kernel runs in
float32 on the simulation hot path; a test pins the two to agree). It carries everything the
measurement needs that is *not* in the jit loop:

- numpy float64 EI / coarse-graining / lumpability (same matrix algebra as the kernel),
- two ways to build the micro transition operator ``T`` from a rollout:
    * Track-S (structural): the random walk on the time-averaged emergent affiliation graph W̄,
    * Track-B (behavioral): a coupling operator estimated from discretized agent trajectories,
- a transition-matrix estimator from a discrete state trajectory (for the estimator gate),
- partition tooling: a reference nested scale hierarchy, agglomerative nested hierarchies,
  greedy-modularity emergent community detection,
- the null comparison (privileged-partition test) and Shapley-EI attribution,
- the paper's synthetic reference network (validation fixture) so the whole pipeline can be
  pinned to known numbers.

Conventions: ``T`` is row-stochastic (N,N); ``p`` is the stationary distribution (N,); a
partition is integer ``labels`` (N,) in ``[0,K)`` or a one-hot ``S`` (N,K). EI is in bits, with
the intervention distribution taken as the chain's own stationary occupancy ``p`` (a
dynamics-derived intervention, not a uniform max-entropy stipulation).
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Dict, List, Optional, Sequence

import numpy as np

_TINY = 1e-300


# =============================================================================
# EI core (numpy float64) — same matrix algebra as effective_information.py
# =============================================================================

def stationary(T: np.ndarray, iters: int = 20000, tol: float = 1e-14) -> np.ndarray:
    """Stationary distribution p with pT = p (power iteration with tolerance)."""
    n = T.shape[0]
    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        pn = p @ T
        if np.abs(pn - p).sum() < tol:
            break
        p = pn
    return p / p.sum()


def partition_to_S(labels: Sequence[int], K: Optional[int] = None) -> np.ndarray:
    """One-hot partition matrix S (N,K) from integer block labels."""
    labels = np.asarray(labels, dtype=int)
    if K is None:
        K = int(labels.max()) + 1
    return np.eye(K)[labels]


def macro_tpm(T: np.ndarray, p: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Stationary-weighted coarse-grained TPM  M = D^-1 Sᵀ diag(p) T S  (K,K)."""
    Pa = S.T @ p                                   # (K,)
    num = S.T @ (p[:, None] * T) @ S               # (K,K)
    with np.errstate(divide="ignore", invalid="ignore"):
        Dinv = np.where(Pa > 0, 1.0 / Pa, 0.0)
    return Dinv[:, None] * num


def ei_bits(M: np.ndarray, intervention: Optional[np.ndarray] = None) -> float:
    """Effective information of row-stochastic M (bits) under an intervention distribution ``w``:

        EI = Σ_a w_a KL(M[a] ‖ M̄_w),   M̄_w = Σ_a w_a M[a].

    ``intervention=None`` → uniform ``w=1/K`` (Hoel max-entropy; matches the reference numbers).
    Pass the normalised macro stationary occupancy (see :func:`macro_stationary`) for the
    dynamics-derived intervention. The stationary weighting that builds ``M`` in
    :func:`macro_tpm` is a *separate* operation from this intervention distribution.
    """
    K = M.shape[0]
    if intervention is None:
        w = np.full(K, 1.0 / K)
    else:
        w = np.asarray(intervention, dtype=float)
        w = w / w.sum()
    avg = w @ M
    avg_safe = np.maximum(avg, _TINY)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_rows = np.where(M > 0, M * np.log2(np.where(M > 0, M, 1.0) / avg_safe), 0.0).sum(axis=1)
    return float((w * kl_rows).sum())


def macro_stationary(p: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Normalised macro block masses ``Pa = Sᵀp`` — the dynamics-derived intervention weights."""
    Pa = S.T @ p
    return Pa / Pa.sum()


def _row_entropy_bits(M: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(M > 0, -M * np.log2(np.where(M > 0, M, 1.0)), 0.0)
    return terms.sum(axis=1)


def det_deg_bits(M: np.ndarray):
    """(determinism, degeneracy) in bits; EI = det - deg."""
    K = M.shape[0]
    avg = M.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        H_avg = float(np.where(avg > 0, -avg * np.log2(np.where(avg > 0, avg, 1.0)), 0.0).sum())
    det = float(np.log2(K) - _row_entropy_bits(M).mean())
    deg = float(np.log2(K) - H_avg)
    return det, deg


def leak(T: np.ndarray, p: np.ndarray, S: np.ndarray) -> float:
    """Kemeny-Snell lumpability leak: stationary-weighted mean std of block-exit probs."""
    Pa = S.T @ p
    with np.errstate(divide="ignore", invalid="ignore"):
        Dinv = np.where(Pa > 0, 1.0 / Pa, 0.0)
    X = T @ S                                       # (N,K) exit prob to each block
    M = Dinv[:, None] * (S.T @ (p[:, None] * X))
    E2 = Dinv[:, None] * (S.T @ (p[:, None] * (X * X)))
    var = np.clip(E2 - M * M, 0.0, None)
    K = S.shape[1]
    return float(np.sqrt(var).sum() / (K * K))


def ei_of_labels(T: np.ndarray, p: np.ndarray, labels: Sequence[int], K=None,
                 intervention=None) -> float:
    """EI (bits) of the coarse-graining by ``labels``. ``intervention``: ``None`` = uniform
    max-entropy (Hoel); ``"stationary"`` = dynamics-derived (the partition's macro stationary
    occupancy); or an explicit weight vector over the K macro states."""
    S = partition_to_S(labels, K)
    M = macro_tpm(T, p, S)
    if intervention == "stationary":
        w = macro_stationary(p, S)
    else:
        w = intervention
    return ei_bits(M, intervention=w)


def leak_of_labels(T: np.ndarray, p: np.ndarray, labels: Sequence[int], K=None) -> float:
    return leak(T, p, partition_to_S(labels, K))


# =============================================================================
# Transition-operator construction from a rollout
# =============================================================================

def rw_transition_from_W(W: np.ndarray, noise: float = 0.05) -> np.ndarray:
    """Track-S: random walk on a weighted graph, T = (1-noise)·rownorm(W) + noise/N.

    Mirrors the reference ``build_T``: a small uniform indeterminism keeps the chain
    irreducible and supplies the micro indeterminism an EI signal needs.
    """
    n = W.shape[0]
    r = W.sum(axis=1, keepdims=True)
    r = np.where(r == 0, 1.0, r)
    return (1.0 - noise) * (W / r) + noise / n


def coupling_transition_from_trajectories(bins: np.ndarray, n_states: Optional[int] = None,
                                          noise: float = 0.05, lag: int = 0) -> np.ndarray:
    """Track-B: build an N×N firm-coupling random walk from discretized agent trajectories.

    ``bins``: (T, N) integer state of each agent at each step. The coupling C[i,j] is the
    above-chance co-occupancy of agents i and j in the same discrete state (optionally with a
    ``lag`` so C[i,j] reflects i(t) predicting j(t+lag), giving directionality). C is floored at
    0, row-normalized, and mixed with uniform noise → a row-stochastic operator estimated from
    *what agents did*, not from the design graph (so it is not circular).
    """
    bins = np.asarray(bins, dtype=int)
    Tn, N = bins.shape
    if n_states is None:
        n_states = int(bins.max()) + 1
    a = bins[: Tn - lag] if lag > 0 else bins
    b = bins[lag:] if lag > 0 else bins
    steps = a.shape[0]
    # one-hot occupancy per state, then co-occurrence counts
    same = np.zeros((N, N))
    pa = np.zeros((N, n_states))
    pb = np.zeros((N, n_states))
    for s in range(n_states):
        Ia = (a == s).astype(float)   # (steps, N)
        Ib = (b == s).astype(float)
        same += Ia.T @ Ib             # co-occupancy counts in state s
        pa[:, s] = Ia.mean(axis=0)
        pb[:, s] = Ib.mean(axis=0)
    same /= steps                                  # observed co-occupancy prob
    expected = pa @ pb.T                            # chance co-occupancy under independence
    C = np.clip(same - expected, 0.0, None)
    # The diagonal is trivial self co-occupancy (an agent is always in its own state), ~1-Σp² for
    # EVERY agent regardless of coupling. Left in, it dominates the walk with self-loops and makes
    # *uncoupled* data coarse-grain to a near-identity macro TPM (spuriously high EI) — the Track-B
    # artefact the estimator gate exposes. Zero it: coupling means cross-agent co-occupancy only.
    np.fill_diagonal(C, 0.0)
    C += 1e-9                                        # uniform floor for irreducibility
    return rw_transition_from_W(C, noise=noise)


def estimate_tpm(state_traj: np.ndarray, n_states: int, laplace: float = 1.0) -> np.ndarray:
    """Estimate a transition matrix from one discrete state trajectory (Laplace-smoothed counts).

    The classic plug-in estimator used in the estimator-validation gate: feed it trajectories of
    known processes and check it recovers the true EI (consistency) and does not manufacture an
    interior EI peak on i.i.d. / single-scale nulls.
    """
    s = np.asarray(state_traj, dtype=int)
    counts = np.full((n_states, n_states), laplace)
    np.add.at(counts, (s[:-1], s[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


# =============================================================================
# Partitions: reference hierarchy, agglomerative hierarchy, emergent communities
# =============================================================================

def reference_scale_hierarchy(N: int, n_blocks: int):
    """The paper's nested hierarchy micro(N) -> N/2 -> meso(n_blocks) -> 2 -> macro(1).

    Returns list of (label, labels-array). Assumes N divisible by n_blocks and block size even.
    """
    bs = N // n_blocks
    out = []
    out.append((f"micro({N})", np.arange(N)))
    # split each block into two halves -> 2*n_blocks groups
    half = np.zeros(N, dtype=int)
    g = 0
    for b in range(n_blocks):
        base = b * bs
        half[base: base + bs // 2] = g
        half[base + bs // 2: base + bs] = g + 1
        g += 2
    out.append((str(2 * n_blocks), half))
    meso = np.repeat(np.arange(n_blocks), bs)
    out.append((f"meso({n_blocks})", meso))
    two = (np.arange(N) >= (N // 2)).astype(int)
    out.append(("2", two))
    out.append((f"macro(1)", np.zeros(N, dtype=int)))
    return out


def _modularity(A: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity of an undirected weighted graph A under a labelling."""
    m2 = A.sum()
    if m2 == 0:
        return 0.0
    k = A.sum(axis=1)
    Q = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        Q += A[np.ix_(idx, idx)].sum() - (k[idx].sum() ** 2) / m2
    return Q / m2


def emergent_partition(W: np.ndarray, return_modularity: bool = False):
    """Greedy agglomerative modularity-maximizing community detection (Clauset-Newman-Moore-style).

    Operates on the symmetrized weighted graph. Deterministic (no RNG), so the emergent partition
    is reproducible — preferred over stochastic Louvain for a measurement pipeline. Returns the
    labelling at the modularity-maximizing merge level.
    """
    A = (W + W.T) / 2.0
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    labels = np.arange(n)
    best_labels = labels.copy()
    best_Q = _modularity(A, labels)
    active = list(range(n))
    # community membership as sets via current label ids
    while len(set(labels)) > 1:
        comms = list(set(labels))
        best_gain = -np.inf
        best_pair = None
        baseQ = _modularity(A, labels)
        for a, b in combinations(comms, 2):
            trial = labels.copy()
            trial[trial == b] = a
            q = _modularity(A, trial)
            if q - baseQ > best_gain:
                best_gain = q - baseQ
                best_pair = (a, b)
        a, b = best_pair
        labels[labels == b] = a
        Q = _modularity(A, labels)
        if Q > best_Q:
            best_Q = Q
            best_labels = labels.copy()
    # relabel to 0..K-1
    _, best_labels = np.unique(best_labels, return_inverse=True)
    if return_modularity:
        return best_labels, best_Q
    return best_labels


def agglomerative_hierarchy(W: np.ndarray):
    """Average-linkage agglomerative nested hierarchy on similarity W. Returns dict K -> labels."""
    A = (W + W.T) / 2.0
    n = A.shape[0]
    # distance = 1 - normalized similarity
    s = A / (A.max() + 1e-12)
    D = 1.0 - s
    np.fill_diagonal(D, 0.0)
    clusters = {i: [i] for i in range(n)}
    labels = np.arange(n)
    out = {n: labels.copy()}
    cluster_ids = list(range(n))
    while len(cluster_ids) > 1:
        best = (np.inf, None, None)
        for ci, cj in combinations(cluster_ids, 2):
            members_i, members_j = clusters[ci], clusters[cj]
            d = D[np.ix_(members_i, members_j)].mean()
            if d < best[0]:
                best = (d, ci, cj)
        _, ci, cj = best
        clusters[ci] = clusters[ci] + clusters[cj]
        del clusters[cj]
        cluster_ids.remove(cj)
        lab = np.zeros(n, dtype=int)
        for new_id, c in enumerate(cluster_ids):
            lab[clusters[c]] = new_id
        out[len(cluster_ids)] = lab.copy()
    return out


# =============================================================================
# Unsupervised partition recovery (is the structure discovered, not assumed?)
# =============================================================================

def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Contingency count matrix between two integer labellings (relabelled to 0..K-1)."""
    ca = np.unique(np.asarray(a), return_inverse=True)[1]
    cb = np.unique(np.asarray(b), return_inverse=True)[1]
    C = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(C, (ca, cb), 1.0)
    return C


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information (sqrt normalisation) between two labellings, in [0,1]."""
    C = _contingency(a, b)
    N = C.sum()
    Pij = C / N
    pi = Pij.sum(axis=1)
    pj = Pij.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(np.where(Pij > 0, Pij * np.log(Pij / (np.outer(pi, pj) + 1e-300)), 0.0))
        Ha = -np.nansum(np.where(pi > 0, pi * np.log(pi), 0.0))
        Hb = -np.nansum(np.where(pj > 0, pj * np.log(pj), 0.0))
    if Ha <= 0.0 and Hb <= 0.0:
        return 1.0                                  # both trivial (one block) → identical
    if Ha <= 0.0 or Hb <= 0.0:
        return 0.0
    return float(mi / np.sqrt(Ha * Hb))


def _adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two labellings (chance-corrected agreement), ≤ 1."""
    C = _contingency(a, b)
    comb = lambda n: n * (n - 1) / 2.0
    sum_c = comb(C).sum()
    a_sum = comb(C.sum(axis=1)).sum()
    b_sum = comb(C.sum(axis=0)).sum()
    total = comb(C.sum())
    if total == 0:
        return 1.0
    expected = a_sum * b_sum / total
    max_index = 0.5 * (a_sum + b_sum)
    if max_index == expected:
        return 1.0
    return float((sum_c - expected) / (max_index - expected))


def partition_recovery(W: np.ndarray, planted_labels: Sequence[int]) -> Dict[str, float]:
    """How well the *unsupervised* emergent partition recovers the planted sub-communities.

    Runs :func:`emergent_partition` (greedy modularity, **not told the number of blocks**) on the
    emergent affiliation graph ``W`` and scores its agreement with ``planted_labels``. High NMI /
    ARI with the detector blind to ``K`` is the evidence that the institutional structure is
    *discovered* rather than assumed — the distinction between "endogenous" and "recovery of a
    planted partition the gradient was pointed at" that a careful reader will press on.
    """
    detected = emergent_partition(W)
    planted = np.asarray(planted_labels, dtype=int)
    return {
        "n_detected": int(len(np.unique(detected))),
        "n_planted": int(len(np.unique(planted))),
        "nmi": _nmi(detected, planted),
        "ari": _adjusted_rand(detected, planted),
    }


# =============================================================================
# Privileged-partition null test and Shapley-EI attribution
# =============================================================================

def time_shuffle_surrogate(bins: np.ndarray, rng) -> np.ndarray:
    """Independently permute each agent's state trajectory in time. Destroys cross-agent
    above-chance co-occupancy (the coupling the Track-B operator measures) while preserving each
    agent's marginal state distribution — the correct surrogate for testing whether an estimated
    coupling is real or a finite-sample artefact."""
    bins = np.asarray(bins, dtype=int)
    T, N = bins.shape
    out = np.empty_like(bins)
    for j in range(N):
        out[:, j] = bins[rng.permutation(T), j]
    return out


def behavioral_ei_surrogate_test(bins: np.ndarray, n_states: int, n_blocks: int,
                                 n_surr: int = 200, seed: int = 0) -> Dict[str, float]:
    """Honest null for the *behavioral* (Track-B) meso-EI.

    The meso-EI runs the whole pipeline — estimate the coupling operator, pick the K=``n_blocks``
    partition by agglomerative optimisation, score its EI. Comparing that *optimised* partition to
    *random* same-size partitions (``null_compare``) is biased: an optimised cut beats random cuts
    even on pure noise, so it reports the ~100th percentile on i.i.d. data. The correct null runs
    the **identical** pipeline on time-shuffled surrogates that preserve each agent's marginal but
    destroy cross-agent coupling. Returns the real EI, the surrogate mean, and the percentile of
    the real EI within the surrogate distribution (high ⇒ the coupling is real, not estimated noise).
    """
    def pipeline(b):
        Tb = coupling_transition_from_trajectories(b, n_states=n_states)
        p = stationary(Tb)
        labels = agglomerative_hierarchy(Tb)[n_blocks]
        return ei_of_labels(Tb, p, labels, n_blocks)

    real = pipeline(bins)
    rng = np.random.default_rng(seed)
    null = np.array([pipeline(time_shuffle_surrogate(bins, rng)) for _ in range(n_surr)])
    return {
        "ei": float(real),
        "surrogate_mean": float(null.mean()),
        "ei_percentile": float(100.0 * (null < real).mean()),
    }


def null_compare(T: np.ndarray, p: np.ndarray, labels: Sequence[int], n_null: int = 500,
                 seed: int = 123) -> Dict[str, float]:
    """Compare a partition's EI / leak against random same-size partitions (Criterion: privileged).

    Returns the partition's EI and leak, the random mean/best, and percentiles (EI: higher better;
    leak: lower better). This is the structural null used on the reference network; for the
    behavioral claim, use ``surrogate_delta_ei`` which preserves marginals.
    """
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1
    N = T.shape[0]
    ei_t = ei_of_labels(T, p, labels, K)
    leak_t = leak_of_labels(T, p, labels, K)
    rng = np.random.default_rng(seed)
    rand_ei, rand_leak = [], []
    for _ in range(n_null):
        perm = rng.permutation(N)
        lab = np.zeros(N, dtype=int)
        # same block sizes as the target partition
        sizes = np.bincount(labels, minlength=K)
        pos = 0
        for b, sz in enumerate(sizes):
            lab[perm[pos: pos + sz]] = b
            pos += sz
        rand_ei.append(ei_of_labels(T, p, lab, K))
        rand_leak.append(leak_of_labels(T, p, lab, K))
    rand_ei = np.array(rand_ei)
    rand_leak = np.array(rand_leak)
    return {
        "ei": ei_t, "leak": leak_t,
        "rand_ei_mean": float(rand_ei.mean()), "rand_ei_best": float(rand_ei.max()),
        "rand_leak_mean": float(rand_leak.mean()), "rand_leak_best": float(rand_leak.min()),
        "ei_percentile": float(100.0 * (rand_ei < ei_t).mean()),
        "leak_percentile": float(100.0 * (rand_leak > leak_t).mean()),
    }


def shapley_ei(T: np.ndarray, p: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Exact Shapley value of each block's contribution to EI (mixed-resolution TPM).

    v(S) = EI of the TPM where blocks in S are lumped and firms in blocks not in S are left at
    micro resolution; v({}) = 0. Exact enumeration (2^K) — fine for K <= ~12.
    """
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1
    members = [np.where(labels == b)[0] for b in range(K)]

    def v(Sset):
        if not Sset:
            return 0.0
        groups = []
        for b in range(K):
            if b in Sset:
                groups.append(list(members[b]))
            else:
                groups += [[m] for m in members[b]]
        N = T.shape[0]
        S = np.zeros((N, len(groups)))
        for gi, g in enumerate(groups):
            S[g, gi] = 1.0
        return ei_bits(macro_tpm(T, p, S))

    phi = np.zeros(K)
    for A in range(K):
        others = [x for x in range(K) if x != A]
        for s in range(len(others) + 1):
            for Sset in combinations(others, s):
                w = factorial(len(Sset)) * factorial(K - len(Sset) - 1) / factorial(K)
                phi[A] += w * (v(set(Sset) | {A}) - v(set(Sset)))
    return phi


def max_phi_share(phi: np.ndarray) -> float:
    tot = phi.sum()
    return float(phi.max() / tot) if tot > 0 else float("nan")


# =============================================================================
# Interior-max / shelf diagnostics
# =============================================================================

def ei_curve(T: np.ndarray, p: np.ndarray, hierarchy) -> List[float]:
    """EI (bits) along a list of (label, labels-array) partitions, fine -> coarse."""
    return [ei_of_labels(T, p, lab) for _, lab in hierarchy]


def interior_max(curve: Sequence[float]) -> Dict[str, float]:
    """Diagnose an interior maximum of an EI-vs-scale curve.

    Returns the argmax index, whether it is strictly interior, and the prominence
    ΔEI = EI(peak) - max(EI(ends)).
    """
    c = np.asarray(curve, dtype=float)
    k = int(np.argmax(c))
    interior = 0 < k < len(c) - 1
    prominence = float(c[k] - max(c[0], c[-1]))
    return {"argmax": k, "is_interior": bool(interior), "prominence": prominence,
            "peak": float(c[k])}


# =============================================================================
# Reference network (validation fixture) — port of sim/ei_scale_demo.py build_W/build_T
# =============================================================================

def build_reference_W(c: float = 0.0, n_blocks: int = 4, block_size: int = 4,
                      seed: int = 7) -> np.ndarray:
    """The paper's synthetic coevolutionary exposure graph (validation fixture).

    Dense within-block coupling + a directed inter-block cycle + firm heterogeneity; ``c`` is the
    structural capture parameter funnelling inter-block exposure toward block 0. Uses a *fresh*
    default_rng(seed) so build_reference_T(0.0) reproduces the paper's (A)/(B) numbers exactly.
    """
    NB, BS = n_blocks, block_size
    N = NB * BS
    inb, cyc, eps = 0.55, 1.00, 0.01
    rng = np.random.default_rng(seed)
    W = np.full((N, N), eps)
    for i in range(N):
        bi = i // BS
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


def build_reference_T(c: float = 0.0, noise: float = 0.05, **kw) -> np.ndarray:
    return rw_transition_from_W(build_reference_W(c, **kw), noise=noise)
