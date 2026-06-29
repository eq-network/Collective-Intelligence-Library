# Transformations catalog

Atomic, reusable `GraphState -> GraphState` steps — the low-level building blocks a
*mechanism* is composed from. Pick one by name from `REGISTRY` (in `__init__.py`).

**Type function:** `TransformFactory = Config -> Transform`
(`cilib.core.protocols.Transform = Callable[[GraphState], GraphState]`).

**Entries:** `message_passing`, `prediction_market`, `token_budget`, `belief_update`,
`resource`. (Plus the `vectorized_message_passing` primitive — imported directly, not
via the registry, since it operates on bare arrays.)

**Add one:**
1. Write a `make_*(cfg) -> Transform` factory in a module here. Keep it pure and
   vectorized (vmap/einsum) so it runs under `lax.scan`.
2. Decorate the returned step with `@transform(reads=[...], writes=[...])` so
   `compile_pipeline` can order it.
3. Add one line to `REGISTRY`, and a behavioral test.

A transformation does *one* thing. If it elicits-matches-clears-settles, it's a
**mechanism** (`cilib.mechanisms`), not a transformation.
