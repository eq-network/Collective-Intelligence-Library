"""
Influence-preserved score — the benchmark's headline number (v0 definition).

The alpha benchmark scores a defense portfolio by how much *collective human influence*
it preserves relative to the undefended baseline of the same scenario (docs/alpha-plan.md,
"Metrics catalog additions"). Each environment supplies its own influence readout — a
per-seed scalar in [0, 1] measuring how strongly outcomes track what the human principals
asked for (e.g. ``governed_commons``'s ``influence_fidelity``). This module holds the
environment-agnostic half: the normalization against the baseline.

v0 definition (deliberately simple; documented so it can be argued with and iterated):

    influence_preserved = (mean(treatment) - mean(baseline)) / (1 - mean(baseline))

i.e. the share of the influence *lost in the undefended baseline* that the defense
recovers. 0 = no better than the baseline (the baseline scored against itself), 1 = full
influence restored, negative = the defense made things worse. Undefined (NaN) when the
baseline already preserves everything — there is nothing for a defense to recover.

Pure numpy over per-seed values (same stance as ``bootstrap.py``); feed both arrays from
the SAME seeds so run-to-run noise cancels.
"""
from __future__ import annotations

import numpy as np


def influence_preserved(treatment: np.ndarray, baseline: np.ndarray) -> float:
    """Share of baseline-lost influence recovered by the treatment. See module docstring.

    Args:
        treatment: (S,) per-seed influence readout in [0, 1] under the defense.
        baseline:  (S,) per-seed influence readout in [0, 1] under the undefended baseline,
                   from the same seeds.

    Returns:
        Scalar float; 0 for baseline-vs-itself, 1 for full restoration, negative if the
        treatment scores below the baseline, NaN if the baseline left nothing to recover.
    """
    t = float(np.mean(np.asarray(treatment, dtype=float)))
    b = float(np.mean(np.asarray(baseline, dtype=float)))
    if b >= 1.0 - 1e-9:
        return float("nan")
    return (t - b) / (1.0 - b)


# metric identity (cilib.metrics.identity), declared by attribute because this
# module imports nothing from the library. Not a share: negative when the
# treatment scores below the baseline.
influence_preserved.kind = "index"
influence_preserved.version = 1
