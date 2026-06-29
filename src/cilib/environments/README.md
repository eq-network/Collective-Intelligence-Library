# Environments catalog

Reusable, governance-agnostic simulation substrates. Pick one by name from `REGISTRY`
(in `__init__.py`); `make_env` / `list_envs` are thin conveniences over it.

**Type function:** `EnvFactory = (**cfg) -> EnvSpec` (see `spec.py`). An `EnvSpec` is a
fully-specified, runnable environment: `config`, `init_fn`, `round_fn`, `trace_fn`,
`metrics`, with `run` / `run_batch` / `evaluate` helpers.

**Entries:** `commons_harvest` (spatial tragedy-of-the-commons, Melting-Pot idiom).

```python
from cilib.environments import make_env, list_envs
env = make_env("commons_harvest", n_agents=16, grid=(18, 18))
finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=64, n_steps=1000)
scores = env.evaluate(traces)        # GovSim metric suite
```

**Add one:**
1. Implement the `EnvSpec` fields in a subpackage (config / state / dynamics / metrics).
2. Define a `build_<name>(**cfg) -> EnvSpec` builder; import it in `__init__.py`.
3. Add one line to `REGISTRY`: `"<name>": build_<name>`; add a substrate test.
