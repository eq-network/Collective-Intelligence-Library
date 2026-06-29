"""
Estimator-validation gate for the causal-emergence pipeline.

The load-bearing demand a rigorous reader makes (and that the paper promises in its experimental
definition but must actually run): before trusting any commons effective-information number, show
that the Track-B estimator

  (i) DETECTS real structure when it exists  (positive controls), and
  (ii) does NOT manufacture a spurious interior EI-vs-scale peak on i.i.d. / single-scale nulls at
       the rollout / bin / state-space sizes the experiment uses  (negative control).

Finite-sample plug-in entropy/MI estimators are biased and can fabricate exactly the inverted-U
the theory predicts. This module builds null and structured trajectories at the experiment's
sizes, runs the same offline pipeline (``coupling_transition_from_trajectories`` → agglomerative
hierarchy → ``ei_curve`` / ``null_compare``), and reports whether the estimator passes the gate.

Run as a script for a quick report:  ``python -m cilib.analysis.validate_estimator``
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from cilib.analysis import causal_emergence as ce


# ----------------------------------------------------------------------------- null / structured data

def iid_trajectories(steps: int, n_agents: int, n_states: int, seed: int = 0) -> np.ndarray:
    """Pure i.i.d. null: every agent draws a fresh uniform state each step. No structure at any
    scale — the estimator must return ~0 privilege and no interior peak."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_states, size=(steps, n_agents))


def block_trajectories(steps: int, n_agents: int, n_blocks: int, n_states: int,
                       flip: float = 0.15, walk: float = 0.2,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Single real scale: agents in a block share a block state that does a lazy random walk; each
    step an agent copies its block state w.p. ``1-flip`` else acts i.i.d. Structure lives at the
    block scale only — a positive control for ``n_blocks`` and a check that no *finer* spurious
    scale is privileged. Returns ``(bins (steps,N), planted_labels (N,))``."""
    rng = np.random.default_rng(seed)
    block = np.repeat(np.arange(n_blocks), n_agents // n_blocks)
    bs = rng.integers(0, n_states, size=n_blocks)
    out = np.empty((steps, n_agents), dtype=int)
    for t in range(steps):
        move = rng.random(n_blocks) < walk
        bs = np.where(move, rng.integers(0, n_states, n_blocks), bs)
        copy = rng.random(n_agents) > flip
        out[t] = np.where(copy, bs[block], rng.integers(0, n_states, n_agents))
    return out, block


# ----------------------------------------------------------------------------- diagnostics

def scale_curve(bins: np.ndarray, n_states: int):
    """Build the Track-B operator from ``bins`` and return ``(T, p, curve)`` where ``curve`` is the
    list of ``(label, EI-bits)`` over the agglomerative hierarchy, fine → coarse."""
    T = ce.coupling_transition_from_trajectories(bins, n_states=n_states)
    p = ce.stationary(T)
    hier = ce.agglomerative_hierarchy(T)                  # {K: labels}
    order = sorted(hier.keys(), reverse=True)             # fine (large K) -> coarse (small K)
    curve = [(str(K), ce.ei_of_labels(T, p, hier[K], K)) for K in order]
    return T, p, curve


def spurious_peak_prominence(bins: np.ndarray, n_states: int) -> float:
    """Prominence of any *interior* maximum of the EI-vs-scale curve (bits). ~0 on a structureless
    null; large only if the estimator fabricates an intermediate privileged scale."""
    _, _, curve = scale_curve(bins, n_states)
    return ce.interior_max([v for _, v in curve])["prominence"]


def surrogate_percentile(bins: np.ndarray, n_states: int, n_blocks: int,
                         n_surr: int = 150, seed: int = 7) -> float:
    """EI percentile of the estimated/optimised K=n_blocks partition against the *surrogate* null —
    the same pipeline run on marginal-preserving time-shuffles. The honest test for an optimised
    partition (a random-partition null would report ~100th percentile even on noise)."""
    return ce.behavioral_ei_surrogate_test(
        bins, n_states=n_states, n_blocks=n_blocks, n_surr=n_surr, seed=seed)["ei_percentile"]


# ----------------------------------------------------------------------------- the gate

def run_gate(steps: int = 200, n_agents: int = 16, n_blocks: int = 4, n_states: int = 6,
             peak_tol: float = 0.10, seed: int = 0) -> Dict[str, float]:
    """Run the full gate at the experiment's sizes and return a verdict dict.

    Passes iff: the i.i.d. null shows no spurious interior peak (prominence ≤ ``peak_tol``) and is
    not declared privileged (percentile < 95); a single-scale block structure IS detected (block
    percentile ≥ 95); and the paper's static reference fixture recovers its known privilege
    (institutional percentile ≥ 95). The ``peak_tol`` is in bits, sized for the finite-sample
    regime here.
    """
    iid = iid_trajectories(steps, n_agents, n_states, seed=seed)
    blk, planted = block_trajectories(steps, n_agents, n_blocks, n_states, seed=seed)

    iid_peak = spurious_peak_prominence(iid, n_states)
    iid_pct = surrogate_percentile(iid, n_states, n_blocks, seed=seed + 1)
    blk_pct = surrogate_percentile(blk, n_states, n_blocks, seed=seed + 2)

    # static reference fixture (no estimation; pins the privileged-partition result)
    Tref = ce.build_reference_T(0.0)
    pref = ce.stationary(Tref)
    ref = ce.null_compare(Tref, pref, np.repeat(np.arange(4), 4), n_null=300, seed=7)

    out = {
        "iid_peak_prominence": float(iid_peak),
        "iid_ei_percentile": float(iid_pct),
        "block_ei_percentile": float(blk_pct),
        "reference_ei_percentile": float(ref["ei_percentile"]),
        "block_recovery_nmi": float(ce._nmi(ce.emergent_partition(
            ce.coupling_transition_from_trajectories(blk, n_states=n_states)), planted)),
    }
    out["passed"] = bool(
        out["iid_peak_prominence"] <= peak_tol
        and out["iid_ei_percentile"] < 95.0
        and out["block_ei_percentile"] >= 95.0
        and out["reference_ei_percentile"] >= 95.0
    )
    return out


def main():
    r = run_gate()
    print("Estimator-validation gate")
    print("-" * 48)
    for k, v in r.items():
        print(f"  {k:24s} {v}")
    print("-" * 48)
    print("  VERDICT:", "PASS" if r["passed"] else "FAIL")


if __name__ == "__main__":
    main()
