# Mechanisms catalog

Composed institutions — markets, networks, democracies — expressed as typed
transforms. A mechanism is internally a composition of `transformations` (e.g. a
market is `elicit -> match -> clear -> settle`). Pick one by name from `REGISTRY`.

**Type function:** `TransformFactory = Config -> Transform`. The returned Transform
**declares its `.reads` / `.writes`** via `@transform`, so `compile_pipeline` derives
execution order from effects. Entries within a *family* keep **disjoint write sets**
so `parallel(market, network)` is always valid.

**Families (Plan 2):** `market` · `network` · `democracy`.
**Entries today:** `market`. The variants — double-auction, sealed-bid, trust-weighted
/ gossip networks, direct / liquid / representative democracy — are the first
open-source follow-ups, each a new file + one `REGISTRY` line.

**Add one:**
1. Compose `transformations` into a `make_<mech>(cfg) -> Transform` factory.
2. `@transform(reads=[...], writes=[...])` with writes disjoint from its family siblings.
3. Add one line to `REGISTRY`; add a swap test (it composes where its siblings do).

**Swap test (the payoff):** replacing one family member with another in a pipeline
must keep the pipeline compiling — same read/write contract, different dynamics.
