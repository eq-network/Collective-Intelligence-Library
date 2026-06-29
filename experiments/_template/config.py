"""Experiment config — frozen parameters + the swept axes.

Copy this folder (``experiments/_template/``) to start a study. Keep *all* knobs here so a
run is fully described by one ``ExperimentConfig`` value; the swept axis is just a field
holding several values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ExperimentConfig:
    governance: str = "endogenous"                  # which paradigm preset
    heterogeneities: Sequence[float] = (0.0, 1.0, 2.0)   # the swept axis
    n_seeds: int = 8                                # vmap'd inside run_batch
    T: int = 120                                    # steps per rollout
    seed: int = 0
