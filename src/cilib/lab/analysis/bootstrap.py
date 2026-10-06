"""
Bootstrap confidence intervals across seeds.

Every headline number from a stochastic simulation should carry an interval, not stand as a bare
point estimate — the repeated demand of any rigorous reader. These helpers turn a vector of
per-seed metric values into a percentile-bootstrap CI, and format ``mean [lo, hi]`` for reporting.

Pure numpy, no SciPy. ``statistic`` must accept an ``axis`` argument (``np.mean``, ``np.median``)
so the bootstrap resamples vectorise.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def bootstrap_ci(values, statistic: Callable = np.mean, n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI of ``statistic`` over per-seed ``values``.

    Returns ``(point, lo, hi)``. With fewer than two values the CI collapses to the point estimate.
    """
    v = np.asarray(values, dtype=float).ravel()
    point = float(statistic(v))
    if v.size < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = statistic(v[idx], axis=1)
    lo = float(np.quantile(boot, (1.0 - ci) / 2.0))
    hi = float(np.quantile(boot, 1.0 - (1.0 - ci) / 2.0))
    return point, lo, hi


def paired_diff_ci(a, b, n_boot: int = 2000, ci: float = 0.95,
                   seed: int = 0) -> Tuple[float, float, float, float]:
    """Percentile-bootstrap CI of ``mean(a - b)`` over paired seeds.

    ``a[i]`` and ``b[i]`` must come from the same seed, as cells of one
    ``EnvSpec.run_grid`` call do, so the seed-to-seed variation both share cancels
    before resampling. Resampling ``a`` and ``b`` separately would throw that pairing
    away and widen the interval. The interval covers seed variation only; comparing
    many cells calls for its own correction.

    Returns ``(point, lo, hi, frac_same_sign)``. The last is the fraction of seeds
    whose own difference is nonzero and has the sign of ``point``, a check that the
    mean is not carried by a few seeds; it is NaN when ``point`` is 0.

    Inputs must be 1-D, of equal length and free of NaN: a ``(G, S)`` array would
    otherwise be pooled into pairs across cells.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"paired samples must be 1-D and of equal length, "
                         f"got shapes {a.shape} and {b.shape}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("paired samples contain NaN")
    d = a - b
    point, lo, hi = bootstrap_ci(d, np.mean, n_boot=n_boot, ci=ci, seed=seed)
    same = (float("nan") if point == 0 or not d.size
            else float(np.mean(np.sign(d) == np.sign(point))))
    return point, lo, hi, same


def summarize(per_seed: Dict[str, np.ndarray], **kw) -> Dict[str, Tuple[float, float, float]]:
    """Map ``{metric_name: (S,) per-seed values}`` → ``{metric_name: (point, lo, hi)}``."""
    return {name: bootstrap_ci(np.asarray(vals), **kw) for name, vals in per_seed.items()}


def format_ci(point: float, lo: float, hi: float, prec: int = 3) -> str:
    """``"0.682 [0.601, 0.751]"`` — point estimate with its bootstrap CI."""
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"
