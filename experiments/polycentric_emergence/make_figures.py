"""
Figures for the polycentric-emergence experiment.

Renders the headline visuals as PNG + PDF and emits a pgfplots coordinate snippet that drops into
paper/main.tex (matching its tikz/pgfplots style). Reuses the experiment's helper functions.

Run:  python -m experiments.polycentric_emergence.make_figures
"""
from __future__ import annotations

import os
import numpy as np
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cilib.paradigms import polycentric as P
from cilib.analysis import causal_emergence as ce
from experiments.polycentric_emergence.run_experiment import (
    run_regime, final_R, survival, mean_fit, behavioral_meso_ei, responsiveness,
    structural_shapley, discretize, T, N_SEEDS,
)

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# paper palette
C = {"atomized": "#55595C", "monocentric": "#C44E52", "fixed_poly": "#DD8452",
     "endogenous": "#4C72B0", "accent": "#8172B3"}
REGIMES = ["atomized", "monocentric", "fixed_poly", "endogenous"]
HETS = [0.0, 0.5, 1.0, 1.5, 2.0]
CAPS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  wrote figures/{name}.png / .pdf")


def main():
    pgf = []  # pgfplots coordinate snippets for main.tex

    # ---- gather regime data at het=1.5 ----
    print("computing regime data (het=1.5)...")
    reg = {}
    for g in REGIMES:
        cfg, finals, trs = run_regime(g, het=1.5)
        me = behavioral_meso_ei(trs, cfg)
        reg[g] = dict(finalR=final_R(trs), surv=survival(trs), fit=mean_fit(trs),
                      ei=me["ei"], resp=responsiveness(trs, finals))

    # ===== Figure 1: governance regimes — survival, fit, and the EI∧fit plane =====
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    xs = np.arange(len(REGIMES))
    ax[0].bar(xs, [reg[g]["surv"] for g in REGIMES], color=[C[g] for g in REGIMES])
    ax[0].set_xticks(xs); ax[0].set_xticklabels(REGIMES, rotation=20, ha="right")
    ax[0].set_ylabel("survival (steps)"); ax[0].set_title("(a) commons survival")
    ax[1].bar(xs, [reg[g]["fit"] for g in REGIMES], color=[C[g] for g in REGIMES])
    ax[1].set_xticks(xs); ax[1].set_xticklabels(REGIMES, rotation=20, ha="right")
    ax[1].set_ylabel("fit (−local regret)"); ax[1].set_title("(b) fit to local conditions")
    for g in REGIMES:
        ax[2].scatter(reg[g]["ei"], reg[g]["fit"], s=120, color=C[g], zorder=3, label=g)
        ax[2].annotate(g, (reg[g]["ei"], reg[g]["fit"]), textcoords="offset points",
                       xytext=(6, 4), fontsize=8)
    ax[2].set_xlabel("meso-EI (institutional scale, bits)")
    ax[2].set_ylabel("fit")
    ax[2].set_title("(c) the EI∧fit plane")
    ax[2].grid(alpha=0.25)
    fig.suptitle("Governance regimes: emergent polycentric co-maintains EI and fit; "
                 "monocentric trades fit; atomized collapses", fontsize=10)
    _save(fig, "fig1_regimes")

    # ===== Figure 2 (HEADLINE): fit gap vs heterogeneity =====
    print("computing heterogeneity sweep...")
    fit_m, fit_e, ei_m, ei_e = [], [], [], []
    for h in HETS:
        _, _, tm = run_regime("monocentric", het=h)
        cfg_e, _, te = run_regime("endogenous", het=h)
        fit_m.append(mean_fit(tm)); fit_e.append(mean_fit(te))
        ei_m.append(behavioral_meso_ei(tm, cfg_e)["ei"]); ei_e.append(behavioral_meso_ei(te, cfg_e)["ei"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(HETS, fit_e, "-o", color=C["endogenous"], label="endogenous (polycentric)")
    ax[0].plot(HETS, fit_m, "-s", color=C["monocentric"], label="monocentric")
    ax[0].fill_between(HETS, fit_m, fit_e, color=C["accent"], alpha=0.18, label="fit gap")
    ax[0].set_xlabel("heterogeneity of sub-communities")
    ax[0].set_ylabel("fit (−local regret)")
    ax[0].set_title("(a) the fit gap opens with heterogeneity")
    ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=0.25)
    ax[1].plot(HETS, ei_e, "-o", color=C["endogenous"], label="endogenous")
    ax[1].plot(HETS, ei_m, "-s", color=C["monocentric"], label="monocentric")
    ax[1].set_xlabel("heterogeneity of sub-communities")
    ax[1].set_ylabel("meso-EI (bits)")
    ax[1].set_title("(b) institutional meso-EI"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
    fig.suptitle("Headline: the welfare cost of uniformity, measured", fontsize=10)
    _save(fig, "fig2_heterogeneity")
    pgf.append("% (Fig 2) fit vs heterogeneity")
    pgf.append("fit_endo: " + " ".join(f"({h:.2f},{v:.4f})" for h, v in zip(HETS, fit_e)))
    pgf.append("fit_mono: " + " ".join(f"({h:.2f},{v:.4f})" for h, v in zip(HETS, fit_m)))
    pgf.append("ei_endo:  " + " ".join(f"({h:.2f},{v:.4f})" for h, v in zip(HETS, ei_e)))
    pgf.append("ei_mono:  " + " ".join(f"({h:.2f},{v:.4f})" for h, v in zip(HETS, ei_m)))

    # ===== Figure 3: load-bearing ablation (monitoring on vs off) =====
    print("computing ablation...")
    cfg_on, f_on, t_on = run_regime("endogenous", het=1.5, monitoring=True)
    cfg_off, f_off, t_off = run_regime("endogenous", het=1.5, monitoring=False)
    metrics = ["survival/200", "fit+1.6", "meso-EI"]
    on_vals = [survival(t_on) / 200, mean_fit(t_on) + 1.6, behavioral_meso_ei(t_on, cfg_on)["ei"]]
    off_vals = [survival(t_off) / 200, mean_fit(t_off) + 1.6, behavioral_meso_ei(t_off, cfg_off)["ei"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(metrics)); w = 0.36
    ax.bar(x - w / 2, on_vals, w, color=C["endogenous"], label="monitoring ON")
    ax.bar(x + w / 2, off_vals, w, color=C["atomized"], label="monitoring OFF (ablation)")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Load-bearing test: enforcement knockout\ncollapses survival, fit AND meso-EI together")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    _save(fig, "fig3_ablation")

    # ===== Figure 4: capture sweep =====
    print("computing capture sweep...")
    cap_fit, cap_share, cap_tot = [], [], []
    for c in CAPS:
        cfg, finals, trs = run_regime("endogenous", het=1.5, capture=c)
        _, share, tot = structural_shapley(finals, cfg)
        cap_fit.append(mean_fit(trs)); cap_share.append(share); cap_tot.append(tot)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(CAPS, cap_fit, "-o", color=C["endogenous"], label="fit")
    ax.set_xlabel("capture  c  (exogenous hub-pull)"); ax.set_ylabel("fit", color=C["endogenous"])
    ax2 = ax.twinx()
    ax2.plot(CAPS, cap_share, "-s", color=C["monocentric"], label="max Shapley-EI share")
    ax2.set_ylabel("max Shapley-EI share", color=C["monocentric"])
    ax.set_title("Capture: an independent dial\nconcentrates causal attribution and erodes fit")
    ax.grid(alpha=0.25)
    _save(fig, "fig4_capture")
    pgf.append("% (Fig 4) capture sweep")
    pgf.append("cap_fit:   " + " ".join(f"({c:.2f},{v:.4f})" for c, v in zip(CAPS, cap_fit)))
    pgf.append("cap_share: " + " ".join(f"({c:.2f},{v:.4f})" for c, v in zip(CAPS, cap_share)))

    # ===== Figure 5: emergent structure + EI-vs-scale (from the simulation) =====
    print("computing emergent structure...")
    cfg, finals, trs = run_regime("endogenous", het=1.5)
    Wbar = np.asarray(finals.global_attrs[P.AFFIL_SUM]).mean(axis=0) / T
    bins = discretize(trs, cfg.n_levels)
    Tb = ce.coupling_transition_from_trajectories(bins, n_states=cfg.n_levels)
    p = ce.stationary(Tb)
    hier = ce.agglomerative_hierarchy(Tb)
    Ks = [k for k in (cfg.n_agents, 8, cfg.n_blocks, 2, 1) if k in hier]
    ei_scale = [ce.ei_of_labels(Tb, p, hier[k]) for k in Ks]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    im = ax[0].imshow(Wbar, cmap="magma")
    ax[0].set_title("(a) emergent affiliation  W̄  (block-structured)")
    ax[0].set_xlabel("agent j"); ax[0].set_ylabel("agent i")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].plot(range(len(Ks)), ei_scale, "-o", color=C["accent"])
    ax[1].set_xticks(range(len(Ks)))
    ax[1].set_xticklabels([("meso" if k == cfg.n_blocks else str(k)) for k in Ks])
    ax[1].set_xlabel("coarse-graining scale (K groups, fine → coarse)")
    ax[1].set_ylabel("EI (bits)")
    ax[1].set_title("(b) EI across scales (emergent commons)")
    ax[1].grid(alpha=0.25)
    _save(fig, "fig5_emergent_structure")
    pgf.append(f"% (Fig 5) EI vs scale, emergent commons (x=1..{len(Ks)} fine->coarse)")
    pgf.append("ei_scale: " + " ".join(f"({i+1},{v:.4f})" for i, v in enumerate(ei_scale)))

    # ---- pgfplots snippet ----
    snippet = os.path.join(FIGDIR, "pgfplots_coords.tex")
    with open(snippet, "w", encoding="utf-8") as fh:
        fh.write("% pgfplots coordinates for paper/main.tex (polycentric-emergence experiment)\n")
        fh.write("\n".join(pgf) + "\n")
    print(f"  wrote figures/pgfplots_coords.tex")
    print("done.")


if __name__ == "__main__":
    main()
