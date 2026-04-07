"""
Visualization for the basin stability experiment.

Two publication-ready plots, both reading from CSVs (decoupled from simulation):

Plot A: Final resources (log y) vs adversarial fraction (x) — one line per mechanism
Plot B: Resource trajectories (log y) vs timestep (x) — adversarial fractions as colors

Usage:
    from experiments.basin_stability.plots import (
        plot_resources_vs_adversarial,
        plot_resource_trajectories,
    )
    fig_a = plot_resources_vs_adversarial("results/summary_*.csv")
    fig_b = plot_resource_trajectories("results/trajectory_*.csv", mechanism="pdd")
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List


MECHANISM_COLORS = {"pdd": "#1f77b4", "prd": "#ff7f0e", "pld": "#2ca02c"}
MECHANISM_LABELS = {"pdd": "PDD", "prd": "PRD", "pld": "PLD"}


def plot_resources_vs_adversarial(
    summary_csv: Path,
    output_path: Optional[Path] = None,
    mechanisms: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot A: Mean final resource (log y) vs adversarial fraction (x).

    One line per mechanism with std error bars.
    """
    df = pd.read_csv(summary_csv)
    if mechanisms:
        df = df[df["mechanism"].isin(mechanisms)]

    fig, ax = plt.subplots(figsize=(8, 5))

    for mech in sorted(df["mechanism"].unique()):
        group = df[df["mechanism"] == mech].sort_values("adversarial_fraction")
        color = MECHANISM_COLORS.get(mech, None)
        label = MECHANISM_LABELS.get(mech, mech.upper())

        ax.errorbar(
            group["adversarial_fraction"],
            group["mean_resource_level"],
            yerr=group["std_resource_level"],
            marker="o", label=label, color=color,
            capsize=3, linewidth=2, markersize=6,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Adversarial Fraction", fontsize=12)
    ax.set_ylabel("Final Resource Level (log scale)", fontsize=12)
    ax.set_title("Resource Survival by Mechanism", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(-0.02, 0.62)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


def plot_resource_trajectories(
    trajectory_csv: Path,
    mechanism: str = "pdd",
    output_path: Optional[Path] = None,
    adversarial_fractions: Optional[List[float]] = None,
) -> plt.Figure:
    """Plot B: Resource level (log y) vs timestep (x).

    Different adversarial fractions as colored lines.
    Shows mean trajectory with shaded +/- 1 std band.
    """
    df = pd.read_csv(trajectory_csv)
    df = df[df["mechanism"] == mechanism]
    if adversarial_fractions:
        df = df[df["adversarial_fraction"].isin(adversarial_fractions)]

    fig, ax = plt.subplots(figsize=(10, 6))

    fracs = sorted(df["adversarial_fraction"].unique())
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(fracs)))

    for frac, color in zip(fracs, colors):
        sub = df[df["adversarial_fraction"] == frac]
        grouped = sub.groupby("step")["resource_level"]
        mean = grouped.mean()
        std = grouped.std().fillna(0)

        ax.plot(mean.index, mean.values, color=color,
                label=f"adv={frac:.0%}", linewidth=1.5)
        ax.fill_between(
            mean.index,
            (mean - std).values,
            (mean + std).values,
            color=color, alpha=0.15,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Resource Level (log scale)", fontsize=12)
    ax.set_title(f"Resource Trajectories — {MECHANISM_LABELS.get(mechanism, mechanism.upper())}",
                 fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig
