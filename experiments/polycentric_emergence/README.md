# Polycentric Emergence — Endogenous Polycentric Governance as Causal Emergence

Worked experiment for the paper *"Where Does Requisite Variety Come From?"*. Replicates the
canonical multi-agent-systems result — **decentralized/emergent governance sustains a common-pool
resource that unenforced play collapses** (Perolat & Leibo 2017; Ren et al. 2025) — inside the
CI-Library JAX commons, then **re-describes it through three lenses** (effective information,
complexity, active inference). See `research/model-spec-v2.md` and
`research/polycentric-governance-prior-work.md`, and the paper claims rewrite
`…/requisite-variety-emergence/claims-v2-polycentric-complexity.md`.

## Run
```
python -m experiments.polycentric_emergence.run_experiment
```
Output is printed and saved to `results.txt`.

## What it shows (latest run, N=16 agents / 4 sub-communities / T=200 / 12 seeds)

| regime | final R | survival | fit | meso-EI | responsiveness |
|---|---|---|---|---|---|
| atomized | 0 | 7 | −1.52 | 0.00 | 0.00 |
| monocentric | 146 | 200 | −1.06 | low | ~0.17 |
| fixed_poly | 144 | 200 | −0.25 | 1.52 (100th pct) | 1.00 |
| endogenous | 144 | 200 | −0.23 | 1.51 (100th pct) | 1.00 |

- **MAS result reproduced:** atomized (no institution, no enforcement) collapses; governed
  regimes survive.
- **Headline (EI∧fit):** the endogenous−monocentric **fit gap grows monotonically with
  heterogeneity** (+0.04 → +0.50 → +0.96 at het 0/1/2) — the welfare cost of uniformity (Oates),
  measured. Endogenous **co-maintains** a causally-privileged institutional meso-scale *and*
  fit *and* responsiveness; monocentric is one calcified setpoint (high control, ~0
  responsiveness, low fit).
- **Load-bearing test:** removing monitoring (policer ablation) collapses survival, fit, and the
  meso-structure together — EI moves when the substantive mechanism is corrupted (not mathiness).
- **Capture:** an exogenous hub-pull `c` concentrates the Shapley-EI attribution and erodes fit
  while the commons can still survive — capture is a measured consequence, not a definition.

## How to read it honestly (epistemic status: Stage 1–2)
- This is a **demonstration that the model produces the predicted signatures**, not calibrated
  empirical science. Resource parameters were chosen so tragedy is the default (atomized
  collapses) while fit-respecting enforced quotas are sustainable (see the note in `schema.py`).
- The **robust discriminators** are **fit**, the **fit-gap vs heterogeneity**, and
  **responsiveness** (corr of harvest with local ideal): monocentric ≈ 0, endogenous ≈ 1.
- **meso-EI** (Track-B, estimated from harvest trajectories) cleanly separates endogenous (real
  4-block structure) from the degenerate/collapsed cases; treat the monocentric meso-EI number
  with care — the behavioral coupling estimator can pick up theta-correlated defection noise, so
  lead with fit/responsiveness when contrasting monocentric vs endogenous.
- Rigor caveats carried from the paper claims (anchor "the right scale" in
  sufficiency/lumpability, dynamics-derived intervention, epistemic-not-ontological emergence)
  apply to any stronger claim built on this.

## Layout
- `engine/paradigms/polycentric/` — the environment (schema, primitives, transforms, agents,
  environment presets, tests).
- `engine/analysis/effective_information.py` — JAX EI kernel (pinned to the paper reference).
- `engine/analysis/causal_emergence.py` — offline pipeline (T construction, estimator, partitions,
  null test, Shapley-EI) + the synthetic reference fixture.
- `run_experiment.py` — this experiment.
