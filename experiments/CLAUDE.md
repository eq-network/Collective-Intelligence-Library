# Experiment conventions

Experiments consume `cilib`; they are not part of the installed package. Copy
`_template/` (`config.py` · `run.py` · `figures.py` · `README.md`); the study list
is in `README.md` here.

- **Figures regenerate from committed results only.** `figures.py` reads
  `results.json` (or the study's exported artifacts); nothing hand-drawn, no
  numbers typed into figure code. A figure you cannot regenerate does not ship.
- **Phase diagrams over trajectories.** Headline claims live on swept planes with
  regime boundaries; single trajectories are illustrations.
- **Comparative claims: paired seeds (common random numbers) and a bootstrap CI.**
  A claim is resolved (CI excludes zero) or reported as unresolved — no magic seed
  counts, no point estimates traveling without error bars.
- **Raw trajectories stay out of git** (`results/` is gitignored); manifests and
  scalar series are small and belong in. Write them through `cilib.runs`: one
  record dir per run, manifest last with checksums, never an ad-hoc JSON.
- **Mechanisms attach as (mechanism, config, ScheduleSpec) triples.** Timing is
  part of the experimental design, never the mechanism's code.
- Prose in study READMEs follows `docs/CLAUDE.md`.
