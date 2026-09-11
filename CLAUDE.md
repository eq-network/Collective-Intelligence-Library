# CLAUDE.md — router for coding agents

Collective Intelligence Library ("CI Lib"): a JAX-native framework for composable
multi-agent simulation. Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it's the
pattern map. This file routes; local rules live next to the work.

```bash
pip install -e .        # editable; required so `import cilib` resolves
python -m pytest -q     # the safety net — keep it green
```

The library installs as `cilib` (distribution: `collective-intelligence-library`).
Always `from cilib.core import ...`; there is **no** top-level `core`/`engine`
package — that was the pre-`src/` layout.

## Working ethos — capability first, cold engineering

*Set 2026-08-14, after a design review drifted into defensibility theater
(clause lists, seed floors, admission gates). This section outranks any instinct
to armor the work. We are trying to build something.*

- **Core capability over defensibility.** The engine's value is what it can
  express and compose: the game-form boundary, the typed transform pipeline,
  conserved ledgers and ports, shared kernels as single objects. Work that adds
  compliance machinery instead of capability is fluff — don't propose it.
  Defensibility is a byproduct of a system that works, never the goal.
- **No arbitrary numbers.** No invented thresholds, seed floors, scores, or
  coined metrics. Proper statistics (CIs, paired seeds) are table stakes, not
  clauses to negotiate. A number is standard practice, or measured with
  provenance, or it does not appear.
- **Say "we don't understand this," in place, plainly.** Unknowns are
  first-class; never smooth one over with a proxy metric or a confident
  sentence. An open question written down beats a weak answer.
- **Thorough over fast.** When a result appears, find *why* it appears before it
  circulates — the insularity floor read as a result until someone asked why it
  existed. Trace mechanisms end to end.
- **Red-team by default.** Substantive designs, deposits, and headline results
  get an independent adversarial pass before commit: spawn subagents mandated to
  *break* the thing (`/red-team`), not polish it. External perspectives are
  brought in deliberately, not by accident.
- **Direct prose, everywhere.** To the point, no fluff; the smart-high-schooler
  rule from the papers applies to internal docs too.

## Router

| Task | Put it in / read first |
|---|---|
| what's next, or the status of any thread | `ROADMAP.md` — the single source of truth; a session that commits ends by moving its card there |
| any engine code | `src/cilib/CLAUDE.md`, then the folder's README |
| a decision rule | `src/cilib/agents/` — its README |
| an atomic `state->state` step | `src/cilib/transformations/` — its README |
| a composed institution (market/network/democracy) | `src/cilib/mechanisms/` — its README |
| a runnable substrate | `src/cilib/environments/` — its README |
| a general-purpose in-loop readout | `src/cilib/metrics/` |
| a full model tied to one study/paper | `src/cilib/lab/paradigms/<name>/` — its README (6-part contract) |
| paper-specific offline math | `src/cilib/lab/analysis/` |
| a study / sweep / figures | `experiments/` — `experiments/CLAUDE.md` |
| a design doc, deposit, or any card/README prose | `docs/CLAUDE.md` |
| a WP paper | `docs/paper-style.md` + the paper's vault README |
| web/view work | the boundary section below |

Each catalog is a plain `REGISTRY = {...}` dict in its `__init__.py`. Adding an
entry = factory + one dict line + a behavioral test. See [EXTENDING.md](EXTENDING.md).

**The lab razor:** would we merge and maintain a stranger's PR to this file the way
we'd maintain a library API? No → it goes under `cilib.lab` (research payload, no
stability promise), not a catalog.

Design docs live in `docs/`; each states its own status and history — read the one
the task touches, not all of them. The two that constrain code:
[docs/game-boundary-design.md](docs/game-boundary-design.md) (frozen `GameSpec`
boundary) and [docs/ledger-design.md](docs/ledger-design.md) (conserved-ledger
coupling grammar). Current direction lives in the project intent (observability
O0/O1 is the next engine move).

## ⟦BOUNDARY⟧ Engine ⇄ view — the two-project contract

*Set 2026-07-30.* The engine (this repo, `src/cilib`) is the
ONLY place model dynamics exist; a behavior not reproducible here from
`(env, config, seed)` does not exist, whatever any web page shows. The view
(eq-network) renders versioned artifacts the engine exported — it never originates
dynamics, metrics, or parameters. Changing an artifact schema is a contract change:
version it and update both sides in one intent, or don't.

The JS dynamics port in `eq-network/apps/playground/src/engine/kernel.js` is
acknowledged debt, and its true size (audited 2026-08-14 against git history) is
**four hand-ported models**, not one:

- `runEconomy` + `runPolitical` — 2026-07-30, the original workbench. Parity vs
  the engine: **unmeasured**.
- the `ledger_society` coupled model — the 2026-08-01 swap (commit 2026-08-02),
  the only port with measured parity (48 seeds/side, ensemble means ± SE;
  [docs/ledger-design.md](docs/ledger-design.md) §8).
- `runPolity` — a WP3 `delegative_polity` port, added 2026-08-07 with the
  showcase scroll; an extension after the never-extend rule, and
  unrecorded here until this audit. Parity: **unmeasured**.

Consequence, stated plainly: the /showcase page's WP-model numbers currently have
unknown fidelity to the engine models the papers are built on. The rule stands —
never extend the kernel further; a new environment, coupling, or metric reaches
the page through the run-record path (O2) or it waits, and O2 is what retires all
four ports at once. A port *swap* requires an explicit maintainer decision plus measured
ensemble parity, a pasted `system_graph()` fixture, and engine-derived series.

## ⟦DEAD END⟧ The GD game

*Decision 2026-08-01: both 2026-07-31 game routes — the precomputed branch
tree and live levers behind a service — were abandoned; full record in
[docs/gd-game-postmortem.md](docs/gd-game-postmortem.md), read it before proposing
anything that calls itself a game. The rule: do not restart either route, and build
nothing until a design starts from what the player does second by second and only
then asks what engine serves it. Leftover code (`mechanisms/families/`,
`mechanisms/interventions.py`, `journey_*` metrics) exists because of the game; its
presence is not a reason to revive either route.*

## Verifying a change

- Any commit → move its card in `ROADMAP.md` (Now / Next / Later / Done).
- Behavior-preserving refactor → `python -m pytest -q` stays green.
- A change to a paradigm's composition → assert the new pipeline is numerically
  identical to the old one for a fixed seed before deleting the old path.
- A new catalog entry → a behavioral test asserting the *mechanism* (direction /
  ordering), not bit-exact numbers.
- Test counts are internal only — never cite "N tests passing" in any external
  communication.
