# CLAUDE.md — orientation for coding agents

Collective Intelligence Library ("CI Lib"): a JAX-native framework for composable
multi-agent simulation. Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it's the
pattern map. This file is the working contract.

## Import root

The library installs as `cilib` (distribution: `collective-intelligence-library`).
Always `from cilib.core import ...`, `from cilib.mechanisms import ...`, etc. There is
**no** top-level `core`/`engine` package anymore — that was the pre-`src/` layout.

```bash
pip install -e .        # editable; required so `import cilib` resolves
python -m pytest -q     # the safety net — keep it green
```

## Where does X go?

| You're adding… | Put it in… | Follow |
|---|---|---|
| a decision rule | `src/cilib/agents/` | `agents/README.md` |
| an atomic `state->state` step | `src/cilib/transformations/` | `transformations/README.md` |
| a composed institution (market/network/democracy) | `src/cilib/mechanisms/` | `mechanisms/README.md` |
| a runnable substrate | `src/cilib/environments/` | `environments/README.md` |
| a full model (agents+world+dynamics) | `src/cilib/paradigms/<name>/` | `paradigms/README.md` (6-part contract) |
| an offline metric on trajectories | `src/cilib/analysis/` or `src/cilib/metrics/` | — |
| a study / sweep | `experiments/<name>/` | `experiments/_template/` |

Each catalog is a plain `REGISTRY = {...}` dict in its `__init__.py`. Adding an entry
= write the factory + add one dict line + a behavioral test. See [EXTENDING.md](EXTENDING.md).

## Load-bearing conventions

- **State lives in `GraphState`, not in objects.** Agents are pure factories; the
  evolving arrays live in `node_attrs` / `adj_matrices`. Don't add stateful classes.
- **Static config is closed over by transform factories, never stored in
  `global_attrs`.** `global_attrs` is static pytree aux — putting swept/per-step data
  there forces recompiles. `GraphState` carries only *evolving* arrays.
- **No data-dependent Python control flow inside transforms.** Replace `if traced:`
  with `jnp.where` / `lax.cond` (e.g. `core.category.gated`) so it traces under `lax.scan`.
- **Pure tier vs eager tier.** Pure (`core.scan`) is the default and the only tier
  that `vmap`s over seeds. Use the eager tier (`core.time`) *only* for genuinely
  effectful agents (LLM/HTTP).
- **Mechanisms/transformations declare `.reads` / `.writes`** via `@transform` so
  `compile_pipeline` can derive order. Same-family mechanisms keep disjoint writes.
- **Simplicity is a hard requirement here.** Prefer short, inspectable code and
  deletion over new abstraction. A catalog you can read in one screen beats a clever
  registry framework.

## Verifying a change

- Behavior-preserving refactor → `python -m pytest -q` must stay green (currently 65).
- A change to a paradigm's composition → assert the new pipeline is numerically
  identical to the old one for a fixed seed before deleting the old path.
- A new catalog entry → a behavioral test asserting the *mechanism* (direction /
  ordering), not bit-exact numbers.
