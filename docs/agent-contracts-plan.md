<!-- Plan recorded 2026-08-14; moved into the repo 2026-09-11. Not yet implemented.
     Card: ROADMAP.md, Next, "Agent type contracts". -->

# Standardized agent type contracts for CI Lib (RL / active inference / LLM / hard-coded)

## Context

CI Lib's catalogs share one type-function pattern (`Config -> TypedCallable`, contracts in `core/protocols.py`), but agents only have contracts for two of the four kinds we run: `Policy` (`(obs, key) -> action`, the hard-coded rules) and `PureAgent` (`round_fn() -> (state,t,key) -> state`, active inference in `lab/paradigms/`). The other two are ghosts: RL learning exists only as inlined ad-hoc code (example 03, `experiments/basin_stability`), and the "LLMAgent HTTP port" is referenced in three docstrings but was never built. The frozen game-boundary doc (`docs/game-boundary-design.md`) explicitly deferred both: "Learning-policy runner (params in the scan carry; who owns the update rule — one design pass when the learning delegate lands)." This is that design pass.

**The unifying frame:** an agent kind is defined by how it meets the simulation loop. Three closure operations over one `GameSpec` — `close` (Policy), `close_learning` (LearningAgent), `run_eager` (EffectfulAgent) — plus `PureAgent`, which owns its whole round and bypasses the boundary. GameSpec becomes the interface between the pure and eager tiers.

**Why the contracts look like this (derivation from the ordinary agent type):** the conceptual type is the classic one — `Agent : (obs, memory) -> (action, memory)` with an internal update rule. Two mechanical constraints reshape it:

1. *JAX purity.* The loop is a compiled scan over vmapped agents, so memory can't hide inside the object; it becomes an explicit `params` dict threaded through the scan carry (`node_attrs["agent__*"]`). Hence `init`: create the memory with known shapes before the scan starts.
2. *The reward boundary.* One call can't both act and learn — reward only exists after `step_fn`. The single function splits into `act` (pre-step, memory read-only) and `update` (post-step, memory write). Composed over a round, `act` + `update` *are* `(obs, memory) -> (action, memory)`.

The four kinds are then regimes of that one type: **Policy** = memory empty (act only); **LearningAgent** = the full type; **EffectfulAgent** = memory opaque (inside the Python object / behind an API), forfeits the scan and runs eager; **PureAgent** = memory folded into world state, owns the whole round. Each regime needs a different loop to run it, which is why there are four contracts and three closures instead of one Union with optional methods.

Scope decisions (defaults taken; the maintainer has not yet answered the scope question; flag at review):
- LLM: contract + `LLMPolicy` with injectable `complete: prompt -> str` (no HTTP dep; tests inject a fake).
- One exemplar per kind: hard-coded exists; promote Q-learning into the catalog; active inference stays in `lab` (lab razor) but conforms to the core protocol; LLM stub agent.
- RL learning is in-scan online only (the proven example-03 pattern: params in `node_attrs`, update inside the compiled scan). An outer-loop trainer can reuse `update` later — it's pure.

## Step 1 — Protocols in `src/cilib/core/protocols.py`

Add (structural Protocols, no ABCs; `core/agents.py` unchanged):

```python
class Transition(NamedTuple):        # one step, ONE agent, at the game boundary
    obs: jnp.ndarray
    action: jnp.ndarray              # exactly what act() returned, untouched by step_fn
    reward: jnp.ndarray              # read via game.rewards(state) AFTER step_fn
    next_obs: jnp.ndarray            # post-step observation

@runtime_checkable
class LearningAgent(Protocol):       # pure, single-agent, vmappable; hyperparams static on instance
    def init(self, key, obs) -> dict: ...   # FLAT str->array dict (node_attrs is Dict[str, ndarray]);
                                            # shapes from obs.shape + config only, never key values
    def act(self, params, obs, key): ...
    def update(self, params, transition: Transition) -> dict: ...

@runtime_checkable
class EffectfulAgent(Protocol):      # host-side, may do I/O; never inside scan/vmap; no key
    def act(self, obs): ...

Agent = Union[Policy, LearningAgent, PureAgent, EffectfulAgent]
AgentFactory = Callable[..., Agent]  # widened from Callable[..., Policy]
```

Deliberate cuts (record in changelog): no key in `update` (TD/SARSA don't need one; adding later is one call-site change), flat-dict params (revisit with `ravel_pytree` only when a real NN agent lands), single shared REGISTRY (README labels which closure each entry needs).

## Step 2 — `close_learning(game, agent) -> EnvSpec` in `src/cilib/environments/game.py`

Beside `close` (needs GameSpec so it can't be core; takes agent *instances* so it never imports the agents catalog — import DAG intact).

- **Init:** wrapped `init_fn(key)` splits the key, builds the game state, then `params = jax.vmap(agent.init)(jr.split(k_params, N), obs)` and writes each leaf as `node_attrs["agent__" + name]`. `close_learning` calls `agent.init`; nobody else. Build-time probe (`game.init_fn(PRNGKey(0))`, precedent: `build_governed_commons`'s `validate_reads` probe) learns param names and raises on collision with existing `node_attrs`.
- **Round order** (the substantive design answer): `obs = observe_fn(state)` → `actions = vmap(agent.act)(params, obs, keys)` → `state = step_fn(state, actions, k_step)` → `reward = game.rewards(state)` (**post-step**, so `last_reward` exists and includes sanction/taxation mechanisms) → `next_obs = observe_fn(state)` → `params = vmap(agent.update)(params, Transition(...))` → write params back. `next_obs` is recomputed next round; accepted double-compute of a cheap pure view.
- Export from `environments/__init__.py`.
- **Tests** (`environments/tests/test_game.py`, extend the existing toy-GameSpec harness): runs under `.run`/`.run_batch`; `agent__*` params change from init; a stub agent whose `update` accumulates reward matches analytic cumulative reward (pins the reward timing); collision guard raises.

## Step 3 — Q-learning exemplar `src/cilib/agents/qlearning.py`

`QLearningAgent(actions, epsilon, alpha, gamma)` — no default hyperparams (no arbitrary numbers). `init -> {"w": (A, D), "b": (A,)}`; `act` epsilon-greedy returning `self.actions[idx]` (the env-level *value* step_fn expects); `update` TD(0), recovering `idx = jnp.argmax(self.actions == tr.action)` — sound because the action is selected from the static array with no arithmetic and `Transition` carries it untouched (document in both docstrings). `REGISTRY["q_learning"] = QLearningAgent`.

Example 03 and `basin_stability` stay untouched — example 03 inlines deliberately ("never depends on experiments/"); catalog reuse there is deferred, noted in the changelog.

Tests: structural `isinstance(agent, LearningAgent)`; bandit behavioral (rewarded arm becomes greedy); vmap conformance; integration through `close_learning` + `run_batch` with early-vs-late reward improvement.

## Step 4 — `run_eager(game, agent, n_steps, key, trace_fn=None)` in `src/cilib/environments/game.py`

Plain host-Python fold reusing the same GameSpec: per tick `observe → [agent.act(obs[i]) for i in range(N)] → step_fn(state, actions, k_step)`; key handling mirrors `EnvSpec.run`; stack trace leaves along leading axis `n_steps` so `game.metrics` apply unchanged.

Deliberately bypasses `core/time.py`'s `World`/`EventLog` (`apply_event` is a stub) — GameSpec is the tier bridge; reword `core/time.py` docstrings accordingly. Scope cuts documented: one agent instance for all seats (matches `close`'s symmetric vmap semantics); stateful Python agents allowed but forfeit key-reproducibility.

Tests: scripted-agent call-count/state assertions, plus a **tier-equivalence test** — the same deterministic rule run as a Policy via `close(...).run` and as an EffectfulAgent via `run_eager` on a deterministic-step toy game yields identical trajectories.

## Step 5 — `LLMPolicy` in `src/cilib/agents/llm.py`

`LLMPolicy(complete, format_obs, parse_action)`; `act(obs) = parse_action(complete(format_obs(obs)))`. All three injectable and **required** — no non-arbitrary default exists for formatting/parsing, and malformed-output handling (retry/clamp/default) belongs to the caller's `parse_action`, keeping the library transport-free. `REGISTRY["llm"] = LLMPolicy`; README marks it eager-tier ("run via `run_eager`, not `close`").

Tests: fake `complete` recording prompts; structural `isinstance(…, EffectfulAgent)`; end-to-end scripted-LLM drives a toy game via `run_eager`.

## Step 6 — Cleanup + docs

- Rewrite `agents/learnable.py::LinearPolicy` as an honest fixed-weight Policy `LinearPolicy(weights, bias)` with a two-arg `__call__` — kills the `params=None` silent-re-init hazard; "learnable linear" is now `QLearningAgent`'s job. (Nothing imports the old form beyond the REGISTRY line — verified.)
- Delete `agents/rl_components.py` (duplicate RandomPolicy/LinearPolicy, uncataloged) and empty `agents/factory.py`, `agents/profiles.py`, after a confirming grep.
- Fix the three stale "LLMAgent / engine.agents.llm_agent" docstrings: `core/time.py:13`, `core/scan.py:12`, `lab/paradigms/active_inference/agents.py:9` — point them at `agents/llm.py` + `run_eager`.
- Update `agents/README.md` (four-kinds table: kind → contract → closure → exemplar), `EXTENDING.md` "Add an agent", `ARCHITECTURE.md` catalog table, a dated status line in `docs/game-boundary-design.md` (deferred learning-runner landed), and `docs/changelogs/2026-08-14_agent-contracts.md` (decisions + deferrals: heterogeneous eager seats, update-with-key, example-03 reuse, nested param pytrees).

## Verification

```powershell
python -m pytest -q                                        # green after every step
python examples\03_vmap_seeds_and_learning.py --smoke      # untouched path still runs
```

REPL smoke: `from cilib.agents import REGISTRY`; `isinstance(REGISTRY["q_learning"](...), LearningAgent)`; `isinstance(LLMPolicy(fake, fmt, parse), EffectfulAgent)`. Regression guards: `environments/tests/test_game.py`, the substrate suites that call `close` (`governed_commons`, `compute_economy`, …), `tests/test_examples.py`, `tests/test_suites.py`.

## Risks

- **Flat-dict params** blocks NN policies (nested pytrees) — if an NN agent is imminent, decide the flattening story now instead.
- **`argmax(actions == action)` index recovery** assumes distinct action values and that step_fn never rewrites the recorded action — both hold today; the Transition docstring pins the second.
- Implementation happens in the CI Lib repo, which has its own working ethos (capability first, red-team substantive designs) — this design touches a frozen boundary doc additively (new closures beside `close`, no change to `GameSpec` or `close` itself), so it should not need a re-freeze.
