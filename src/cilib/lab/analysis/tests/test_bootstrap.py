"""Paired-difference bootstrap: the pairing is what narrows the interval."""
import numpy as np
import pytest

from cilib.lab.analysis.bootstrap import bootstrap_ci, paired_diff_ci


def test_pairing_cancels_the_shared_seed_noise():
    rng = np.random.default_rng(0)
    seed_effect = rng.normal(0.0, 1.0, 64)             # shared by both arms
    a = seed_effect + 0.1 + rng.normal(0.0, 0.01, 64)
    b = seed_effect
    point, lo, hi, same = paired_diff_ci(a, b)
    assert point == pytest.approx(np.mean(a - b))
    assert 0.0 < lo < point < hi
    assert same == 1.0
    _, ulo, uhi = bootstrap_ci(a)                      # one arm alone: seed noise dominates
    assert hi - lo < (uhi - ulo) / 10


def test_sign_fraction_counts_disagreeing_seeds():
    point, _, _, same = paired_diff_ci([1.0, 1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0])
    assert point == pytest.approx(0.5)
    assert same == pytest.approx(0.75)


def test_unpaired_sizes_are_rejected():
    with pytest.raises(ValueError, match="paired"):
        paired_diff_ci([1.0, 2.0], [1.0])


def test_sign_fraction_is_undefined_without_an_effect():
    assert np.isnan(paired_diff_ci([1.0, -1.0, 0.0, 0.0], [0.0] * 4)[3])
    assert np.isnan(paired_diff_ci([0.3, 0.5], [0.3, 0.5])[3])


@pytest.mark.parametrize("a,b", [([[1.0, 2.0]], [[1.0, 2.0]]), ([1.0, np.nan], [1.0, 2.0])])
def test_pooled_or_nan_samples_are_rejected(a, b):
    with pytest.raises(ValueError, match="paired samples"):
        paired_diff_ci(a, b)
