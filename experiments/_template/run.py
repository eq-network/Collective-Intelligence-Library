"""Run the sweep, reduce to metrics, save results.

The canonical experiment shape:  build config -> for each sweep point run the paradigm over
seeds (``run_batch`` vmaps the seeds into one compiled program) -> reduce each rollout batch to
scalar metrics -> save. Swap the paradigm (``cilib.paradigms.*``), the metrics, and the sweep
axis for your own study.

    python -m experiments._template.run
"""
from __future__ import annotations

import json
import os

import numpy as np
import jax.random as jr

from cilib.paradigms import polycentric as P

from .config import ExperimentConfig


# --- metrics: reduce a (seeds, T, ...) trajectory batch to scalars -----------

def mean_fit(trs, last: int = 40) -> float:
    return float(np.asarray(trs["fit"])[:, -last:, :].mean())


def final_resource(trs) -> float:
    return float(np.asarray(trs["resource"])[:, -1].mean())


# --- sweep -------------------------------------------------------------------

def run(cfg: ExperimentConfig):
    rows = []
    for het in cfg.heterogeneities:
        pcfg = P.make_config(cfg.governance, heterogeneity=het)
        _, trs = P.run_batch(pcfg, jr.PRNGKey(cfg.seed), n_seeds=cfg.n_seeds, T=cfg.T)
        rows.append({"het": float(het), "fit": mean_fit(trs), "final_R": final_resource(trs)})
    return rows


def main():
    cfg = ExperimentConfig()
    rows = run(cfg)

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)

    for r in rows:
        print(f"het={r['het']:.1f}  fit={r['fit']:+.3f}  final_R={r['final_R']:.1f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
