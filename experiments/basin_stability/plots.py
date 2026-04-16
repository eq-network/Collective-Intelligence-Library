"""
Visualization for the basin stability experiment.

Reads from summary and trajectory CSVs (decoupled from simulation).
Supports the 3x2 factorial design (mechanism x tracking mode).

Usage:
    from experiments.basin_stability.plots import plot_summary, plot_trajectories
    fig = plot_summary("results/summary_*.csv")
    fig = plot_trajectories("results/trajectory_*.csv", mechanism="pdd")
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List


MECHANISM_COLORS = {"pdd": "#1f77b4", "prd": "#ff7f0e", "pld": "#2ca02c"}
TRACKING_STYLES = {"predictive": "-o", "non_predictive": "--s"}

# Predictive = PDD/PRD/PLD, Non-predictive = DD/RD/LD
_LABELS = {
    ("pdd", "predictive"): "PDD",
    ("pdd", "non_predictive"): "DD",
    ("prd", "predictive"): "PRD",
    ("prd", "non_predictive"): "RD",
    ("pld", "predictive"): "PLD",
    ("pld", "non_predictive"): "LD",
}


def _label(mechanism, tracking_mode):
    return _LABELS.get((mechanism, tracking_mode), f"{mechanism.upper()} ({tracking_mode})")


def _style_key(mechanism, tracking_mode):
    return TRACKING_STYLES.get(tracking_mode, "-o"), MECHANISM_COLORS.get(mechanism, None)


def plot_summary(
    summary_csv,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """4-panel summary: selected yield, capture rate, delegation gini, voting entropy."""
    df = pd.read_csv(summary_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Mean selected yield ---
    ax = axes[0, 0]
    for (mech, track), grp in df.groupby(["mechanism", "tracking_mode"]):
        grp = grp.sort_values("adversarial_fraction")
        style, color = _style_key(mech, track)
        ax.plot(grp["adversarial_fraction"], grp["mean_selected_utility"],
                style, label=_label(mech, track),
                color=color, markersize=5,
                alpha=1.0 if track == "predictive" else 0.5)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="break-even")
    ax.set_xlabel("Adversarial Fraction")
    ax.set_ylabel("Mean Selected Yield")
    ax.set_title("Portfolio Quality Selected by Mechanism")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Capture rate (PRD) ---
    ax = axes[0, 1]
    prd = df[df["mechanism"] == "prd"]
    for track, grp in prd.groupby("tracking_mode"):
        grp = grp.sort_values("adversarial_fraction")
        style, color = _style_key("prd", track)
        ax.plot(grp["adversarial_fraction"], grp["mean_capture_rate"],
                style, label=_label("prd", track), color=color, markersize=5,
                alpha=1.0 if track == "predictive" else 0.5)
    fracs = sorted(df["adversarial_fraction"].unique())
    ax.plot(fracs, fracs, ":", color="gray", alpha=0.5, label="proportional")
    ax.set_xlabel("Adversarial Fraction")
    ax.set_ylabel("Adversarial Capture Rate")
    ax.set_title("Representative Capture (PRD)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Delegation Gini (PLD) ---
    ax = axes[1, 0]
    pld = df[df["mechanism"] == "pld"]
    for track, grp in pld.groupby("tracking_mode"):
        grp = grp.sort_values("adversarial_fraction")
        style, color = _style_key("pld", track)
        ax.plot(grp["adversarial_fraction"], grp["mean_delegation_gini"],
                style, label=_label("pld", track), color=color, markersize=5,
                alpha=1.0 if track == "predictive" else 0.5)
    ax.set_xlabel("Adversarial Fraction")
    ax.set_ylabel("Delegation Gini")
    ax.set_title("Delegation Concentration (PLD)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Voting entropy ---
    ax = axes[1, 1]
    for (mech, track), grp in df.groupby(["mechanism", "tracking_mode"]):
        grp = grp.sort_values("adversarial_fraction")
        style, color = _style_key(mech, track)
        ax.plot(grp["adversarial_fraction"], grp["mean_voting_entropy"],
                style, label=_label(mech, track),
                color=color, markersize=5,
                alpha=1.0 if track == "predictive" else 0.5)
    ax.set_xlabel("Adversarial Fraction")
    ax.set_ylabel("Voting Entropy (bits)")
    ax.set_title("Vote Dispersion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns else "?"
    fig.suptitle(f"Basin Stability Experiment — {n_seeds} seeds, T=200", fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_trajectories(
    trajectory_csv,
    mechanism: str = "pdd",
    tracking_mode: str = "predictive",
    metric: str = "resource_level",
    output_path: Optional[Path] = None,
    adversarial_fractions: Optional[List[float]] = None,
) -> plt.Figure:
    """Resource trajectories (log y) vs timestep for one mechanism+tracking combo."""
    df = pd.read_csv(trajectory_csv)
    df = df[(df["mechanism"] == mechanism) & (df["tracking_mode"] == tracking_mode)]
    if adversarial_fractions:
        df = df[df["adversarial_fraction"].isin(adversarial_fractions)]

    fig, ax = plt.subplots(figsize=(10, 6))

    fracs = sorted(df["adversarial_fraction"].unique())
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(fracs)))

    for frac, color in zip(fracs, colors):
        sub = df[df["adversarial_fraction"] == frac]
        grouped = sub.groupby("step")[metric]
        mean = grouped.mean()
        std = grouped.std().fillna(0)

        ax.plot(mean.index, mean.values, color=color,
                label=f"adv={frac:.0%}", linewidth=1.5)
        ax.fill_between(
            mean.index,
            np.clip(mean.values - std.values, 1e-10, None),
            mean.values + std.values,
            color=color, alpha=0.15,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel(f"{metric} (log scale)", fontsize=12)
    label = _label(mechanism, tracking_mode)
    ax.set_title(f"{label} — {metric}", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig
