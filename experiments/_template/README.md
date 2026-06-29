# Experiment template

Copy this folder to start a study. The shape mirrors the paradigm contract — one role per file:

| File | Role |
|---|---|
| `config.py` | frozen `ExperimentConfig`: fixed params + the swept axes (a run = one config value) |
| `run.py` | build config → sweep (run a paradigm over seeds via `run_batch`) → reduce to metrics → save `results.json` |
| `figures.py` | read `results.json` → emit figures (never recompute) |
| `README.md` | the hypothesis + what the sweep varies |

```bash
python -m experiments._template.run        # -> results.json
python -m experiments._template.figures     # -> fit_vs_heterogeneity.png
```

This template sweeps **heterogeneity** for the polycentric paradigm and plots **mean fit**.
To make your own study: swap the paradigm (`cilib.paradigms.*`), the metrics in `run.py`, and
the sweep axis in `config.py`. For a full worked study see `experiments/polycentric_emergence/`.
