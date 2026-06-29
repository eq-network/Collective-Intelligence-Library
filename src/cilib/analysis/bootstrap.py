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


def summarize(per_seed: Dict[str, np.ndarray], **kw) -> Dict[str, Tuple[float, float, float]]:
    """Map ``{metric_name: (S,) per-seed values}`` → ``{metric_name: (point, lo, hi)}``."""
    return {name: bootstrap_ci(np.asarray(vals), **kw) for name, vals in per_seed.items()}


def format_ci(point: float, lo: float, hi: float, prec: int = 3) -> str:
    """``"0.682 [0.601, 0.751]"`` — point estimate with its bootstrap CI."""
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"
