# Contributing

CI Lib grows by **addition, not restructuring**: a new building block is a new file
plus one line in a catalog's `REGISTRY`. The core framework stays stable; the library
gets richer as catalogs fill in.

## Setup

```bash
git clone https://github.com/eq-network/Collective-Intelligence-Library.git
cd Collective-Intelligence-Library
pip install -e .          # editable install; `import cilib` now resolves
python -m pytest -q       # confirm green before you start
```

This is research code evolving in the open, and the codebase will make more sense
after a short conversation than after a long read. If you're interested in the ideas,
the direction, or collaborating — please reach out:
[Jonas Hallgren](https://github.com/spiralling) · Uppsala University · `jonas@eq-network.org`

Looking for something to do? [ROADMAP.md](ROADMAP.md) is the single source of truth
for what comes next. Pick a card from Next or Later and open an issue naming it.

## The contribution path

1. **Pick a catalog** — agents, transformations, mechanisms, environments (or a whole
   paradigm / experiment). See [EXTENDING.md](EXTENDING.md) for the recipe.
2. **Satisfy its type function** — the `Config -> TypedCallable` contract in
   [`src/cilib/core/protocols.py`](src/cilib/core/protocols.py). Conformance is
   structural: match the shape, no base class to inherit.
3. **Register it** — add one line to the catalog's explicit `REGISTRY` dict.
4. **Test it** — a behavioral test asserting the *mechanism* (direction, ordering,
   qualitative phenomenon), not bit-exact numbers. Put it in the catalog's `tests/`.
5. **Keep it small.** Simplicity and inspectability are explicit goals — prefer short,
   readable code over abstraction. If a reviewer can't read your block in one screen,
   it's probably doing too much.

## House style

- State lives in `GraphState`, not in objects; agents are pure factories.
- Static config is closed over by factories, never stored in `global_attrs`.
- No data-dependent Python control flow inside transforms (`jnp.where` / `lax.cond`).
- New mechanisms/transformations declare `.reads` / `.writes` via `@transform`.

See [CLAUDE.md](CLAUDE.md) for the full conventions and [ARCHITECTURE.md](ARCHITECTURE.md)
for how the pieces relate.

## Documentation cadence

Per-change notes go in `docs/changelogs/` as dated Markdown on the day the change
lands; architectural changes get compiled into a typeset monthly roll-up (see
`docs/changelogs/2026-03-30_core-primitives.pdf` for the template). If something in
the docs contradicts the code, **the code is the source of truth** and the docs are
drifting — opening an issue or pinging Jonas is the fastest fix.
