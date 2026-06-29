"""
Tests for the cold-read-driven analysis extensions:

  * dynamics-derived (stationary-weighted) effective information,
  * the estimator-validation gate (i.i.d. / single-scale nulls; no spurious interior peak),
  * the surrogate-trajectory null for the behavioral meso-EI (and the self-mass-diagonal fix it
    exposed in the coupling estimator),
  * unsupervised partition recovery (NMI / ARI vs the planted blocks),
  * bootstrap confidence intervals across seeds.
"""
import numpy as np
import pytest

from cilib.analysis import causal_emergence as ce
from cilib.analysis import validate_estimator as ve
from cilib.analysis import bootstrap as bs
from cilib.analysis.effective_information import ei_bits as jax_ei_bits


# ----------------------------------------------------------------------------- stationary-weighted EI

def _row_stochastic(rows):
    M = np.asarray(rows, dtype=float)
    return M / M.sum(axis=1, keepdims=True)


def test_ei_default_is_uniform_intervention():
    M = _row_stochastic([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    # default (None) must equal an explicit uniform intervention
    assert ce.ei_bits(M) == pytest.approx(ce.ei_bits(M, intervention=np.ones(3)), abs=1e-12)


def test_ei_intervention_changes_value_and_stays_finite():
    M = _row_stochastic([[0.95, 0.05], [0.1, 0.9], [0.5, 0.5]])
    skewed = ce.ei_bits(M, intervention=np.array([0.8, 0.1, 0.1]))
    uniform = ce.ei_bits(M)
    assert np.isfinite(skewed) and skewed >= 0.0
    assert skewed != pytest.approx(uniform, abs=1e-6)   # a non-uniform intervention should move it


def test_ei_jax_numpy_agree():
    M = _row_stochastic([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    assert float(jax_ei_bits(M)) == pytest.approx(ce.ei_bits(M), abs=1e-5)


def test_macro_stationary_normalised():
    p = np.array([0.4, 0.1, 0.3, 0.2])
    S = ce.partition_to_S([0, 0, 1, 1], 2)
    w = ce.macro_stationary(p, S)
    assert w.sum() == pytest.approx(1.0)
    assert w == pytest.approx([0.5, 0.5])           # block masses 0.5/0.5


# ----------------------------------------------------------------------------- partition recovery

def test_partition_recovery_perfect():
    W = np.kron(np.eye(4), np.ones((4, 4)))          # 4 clean disconnected blocks of 4
    planted = np.repeat(np.arange(4), 4)
    r = ce.partition_recovery(W, planted)
    assert r["n_detected"] == 4
    assert r["nmi"] == pytest.approx(1.0, abs=1e-9)
    assert r["ari"] == pytest.approx(1.0, abs=1e-9)


def test_partition_recovery_scores_in_range():
    rng = np.random.default_rng(0)
    W = rng.random((16, 16))
    r = ce.partition_recovery(W, np.repeat(np.arange(4), 4))
    assert 0.0 <= r["nmi"] <= 1.0                    # NMI is in [0,1]
    assert -0.5 <= r["ari"] <= 1.0                   # ARI is chance-corrected: may be slightly < 0
    assert 1 <= r["n_detected"] <= 16


# ----------------------------------------------------------------------------- bootstrap CIs

def test_bootstrap_ci_brackets_mean():
    point, lo, hi = bs.bootstrap_ci(np.arange(1, 21), seed=0)
    assert point == pytest.approx(10.5)
    assert lo < point < hi
    assert lo >= 1.0 and hi <= 20.0


def test_bootstrap_ci_degenerate_cases():
    assert bs.bootstrap_ci([5.0]) == (5.0, 5.0, 5.0)             # single value
    assert bs.bootstrap_ci(np.full(10, 3.0)) == (3.0, 3.0, 3.0)  # zero variance


def test_format_ci():
    assert bs.format_ci(0.682, 0.601, 0.751) == "0.682 [0.601, 0.751]"


# ----------------------------------------------------------------------------- surrogate null + gate

def test_surrogate_null_separates_structure_from_noise():
    blk, _ = ve.block_trajectories(200, 16, 4, 6, seed=1)
    iid = ve.iid_trajectories(200, 16, 6, seed=1)
    blk_res = ce.behavioral_ei_surrogate_test(blk, n_states=6, n_blocks=4, n_surr=80, seed=1)
    iid_res = ce.behavioral_ei_surrogate_test(iid, n_states=6, n_blocks=4, n_surr=80, seed=1)
    # real coupling must out-score its marginal-preserving surrogate; pure noise must not
    assert blk_res["ei_percentile"] >= 90.0
    assert iid_res["ei_percentile"] <= 50.0
    assert blk_res["ei_percentile"] > iid_res["ei_percentile"]


def test_estimator_gate_passes():
    r = ve.run_gate()
    assert r["passed"] is True
    assert r["iid_peak_prominence"] <= 0.10        # no spurious interior peak on i.i.d. null
    assert r["iid_ei_percentile"] < 95.0           # i.i.d. not declared privileged (vs surrogate)
    assert r["block_ei_percentile"] >= 95.0        # real single-scale structure detected
    assert r["reference_ei_percentile"] >= 95.0    # known fixture privilege recovered
