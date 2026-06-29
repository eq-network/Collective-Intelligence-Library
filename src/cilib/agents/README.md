# Agents catalog

Pre-made, swappable decision rules. Pick one by name from `REGISTRY` (in
`__init__.py` — open it to see every entry).

**Type function:** `AgentFactory = Config -> Policy`, where a `Policy` is a callable
`(obs, key) -> action` (`cilib.core.protocols.Policy`). A *pure* agent that threads
no Python state may instead implement `PureAgent` — `round_fn() -> (state, t, key) -> state` —
so it runs inside `core.scan.run_scan` and `vmap`s over seeds.

**Entries:** `random`, `tit_for_tat`, `linear`. (Also present: `rl_components`,
`profiles`, `democracy/` — legacy/effectful, not yet cataloged.)

**Add one:**
1. Write a `Policy` (or `PureAgent`) in a module here.
2. Add one line to `REGISTRY` in `__init__.py`: `"my_agent": MyPolicy`.
3. Add a behavioral test asserting what it does.

The legacy OOP `core.agents.Agent.act()` is *not* the pattern — prefer pure
`Policy`/`PureAgent`. Effectful agents (LLM/HTTP) live on the eager `core.time` tier.
