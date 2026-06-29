"""
Cross-paradigm measurement tier — pure, ``GraphState``-free analysis functions.

These are fed recorded histories / operators (not simulation state) and return scalars,
mirroring the stance of ``engine/environments/commons_metrics.py``. The simulation tier
(``core.scan`` + paradigm transforms) produces trajectories; this tier measures them.

Currently:
- ``effective_information`` — Hoel/Klein–Hoel EI, coarse-graining, lumpability leak (JAX kernel).
- ``causal_emergence`` — offline pipeline (numpy float64): T construction (structural/behavioral),
  TPM estimator, partitions (nested/agglomerative/greedy-modularity), null test, Shapley-EI,
  and the synthetic reference network (validation fixture).
"""
from . import causal_emergence  # noqa: F401  (submodule, used as engine.analysis.causal_emergence)
from .effective_information import (
    stationary,
    partition_to_S,
    macro_tpm,
    coarse_grain,
    ei_bits,
    det_bits,
    deg_bits,
    ei_components,
    leak,
    ei_of_partition,
)

__all__ = [
    "stationary",
    "partition_to_S",
    "macro_tpm",
    "coarse_grain",
    "ei_bits",
    "det_bits",
    "deg_bits",
    "ei_components",
    "leak",
    "ei_of_partition",
]
