"""Figures from saved results — read ``results.json``, plot, never recompute.

Keeping figures separate from the run means you can re-style plots without re-running the
sweep. Run ``run.py`` first.

    python -m experiments._template.figures
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")                # headless-safe
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "results.json")) as fh:
        rows = json.load(fh)

    hets = [r["het"] for r in rows]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(hets, [r["fit"] for r in rows], "o-")
    ax.set_xlabel("heterogeneity")
    ax.set_ylabel("mean fit")
    ax.set_title("template study")
    fig.tight_layout()

    out = os.path.join(here, "fit_vs_heterogeneity.png")
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
