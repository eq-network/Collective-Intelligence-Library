# Roadmap

The single source of truth for what the library does next. Status lines in other
docs, the website roadmap, and any personal boards derive from this file, never the
reverse. A change that lands in the engine ends by moving its card here.

Sections: **Now** (claimed, in progress), **Next** (agreed, in rough order),
**Later** (worth doing, unscheduled), **Done** (newest first, one line each, naming
the record that carries the detail). Cards carry `#tags` so you can filter by area.
To pick one up, open an issue naming the card.

Standing constraints, in full in [CLAUDE.md](CLAUDE.md): JAX/JIT compatibility (no
data-dependent control flow in transforms; static config closed over, never in
`global_attrs`); the trajectory is the memory ceiling, not the state; deletion over
abstraction; static-host-friendly artifacts; `python -m pytest -q` stays green.

## Direction (horizon 2026-Q3)

A durable observability layer over the finished alpha: runs that leave
self-describing, versioned records, and a metric and chart vocabulary stable enough
that a chart built today means the same thing against a run made in six months.
The engine is no longer the bottleneck (reducers made metrics O(1), `system_graph`
made topology self-describing); durability across time is. Until the run-record
path lands, new environments and register entries stay in Later.

**Open decisions**

- **Order of Next.** The horizon says run records (O0 + O1) first. The last working
  session (2026-09-10) asked for computation and agent-definition work first, to
  run larger experiments. O0 + O1 landed 2026-09-11 on a branch (merged
  2026-09-30), and 2026-09-30 Jonas again chose core and compute work next. Not
  formally resolved; the list below keeps the horizon's order.
- **O1 scope.** Resolved 2026-09-11: one writer, `cilib.runs`, with the two fixture
  exporters building on its manifest core and `perf_baseline` sharing only the
  engine block (its output is a timing file, not an address). Still open: what
  `assumption_audits` should leave behind, since it currently prints and nothing
  durable exists to route.
- **AI listening rows.** `ai_insularity` in `ledger_society` is a workaround for
  the frozen AI rows, not a resolution; letting them drift under the attachment
  kernel would make "who the AI attends to" endogenous. Needs its own equivalence
  run (`docs/ledger-design.md` section 9).

## Now

- [ ] **WP4 `delegated_judgment` — read-through and red-team before commit**
  (built 2026-09-15, untracked). Env + `myopic_delegation` policy + 15-rung
  ladder + `experiments/wp4_oversight/` (results.json, figures into the vault
  paper `papers/wp4-oversight/`) + bundle `delegated-judgment-erosion-v1`
  (3 expectations pass; exported; copied to eq-network
  `apps/site/public/lab/runs/`, replayed by `/showcase/judgment`). Owed: Jonas
  read-through of the paper and card; `/red-team` on the model design;
  `%% VERIFY` the memory-written bib; the E4 miss (pooling moves delegation
  too, through the belief) stays reported as found. #wp4 #claim-discipline
- [ ] **`compute_supply`: step-by-step Leontief compute supply chain** (built
  2026-09-23, untracked). Pichler-style orders, input stocks and proportional
  rationing, as a background world component with no agents; stylized uniform
  coefficients, so only the network structure is claimed. Built ahead of O0/O1
  by Jonas's call 2026-09-23, overriding the Direction's "new environments stay
  in Later". Owed: `/red-team` on the model design; a sourced calibration card;
  a check against Pichler's published runs of the upstream order leak and the
  post-shock lock-in, where stocks never refill at zero spare capacity (card
  field 4). #economy #world

## Next

- [ ] **Performance, remaining moves** (`experiments/perf_baseline/README.md`;
  moves 1 and 2 are Done). (a) `run_reduced` / `run_reduced_batch` onto the same
  compile-once path as `run`; they still recompile every call, and the
  `08_streaming_metrics` example that uses them is a quarter of the warm suite.
  (b) Structured or sparse adjacency for the ledger environments: from about 600
  nodes the dense `(N, N)`
  ledgers grow at N^1.6 to N^1.8, and `capital_economy` pays the same because its
  6-sector recipe is stored padded to `(N, N)` and `distribute` rebuilds an `(N, N)`
  one-hot every tick; index the sector block as `(N, S)` first, bit-identity test
  against the current pipeline. Seeds are near free; the 10k-agent milestone is
  blocked on the ledgers, not the scan. #performance #engine
- [ ] **Agent type contracts** — four agent kinds as regimes of one type
  `(obs, memory) -> (action, memory)`: `Policy` (no memory), `LearningAgent`
  (`init` / `act` / `update`, params threaded through the scan carry as
  `node_attrs["agent__*"]`), `EffectfulAgent` (opaque memory, runs eager),
  `PureAgent` (owns its round). Three closures over one `GameSpec`: `close`
  (exists), `close_learning`, `run_eager`. One exemplar per kind: Q-learning
  promoted into the catalog, `LLMPolicy` with an injectable `complete` (no HTTP
  dependency), active inference stays in `lab` but conforms. Cleanup: honest
  fixed-weight `LinearPolicy`, delete the uncataloged duplicates. Additive to the
  frozen boundary. Plan: `docs/agent-contracts-plan.md`. #agent #engine
- [ ] **Clean invariance run** — the transfer-gap finding flips sign under the ledger
  implementation (64 paired seeds, `docs/ledger-design.md` section 9), but the two
  runs changed the defenses as well as the coupling. Build defenses that exist in
  both models, re-run paired, and only then attribute the sign to the coupling.
  Until then the kappa-era transfer-gap numbers stay out of every claim. #ledger
  #claim-discipline
- [ ] **Name the floor on every assumptions card** — one line in slider language per
  scenario: what will not go below what, because of which constant or structure,
  and that removing it goes to zero. The audits have the numbers
  (`experiments/assumption_audits/`). #public
- [ ] **Spectral family, remaining rungs** — only `fiedler_partition_alignment`
  shipped; spectral-gap ratio, current-flow mediation shares, and the
  `spectral_margin` promotion are open (`docs/cultural-register-design.md` section
  8). Spectral lock-in, the one place the build approaches irreversibility, is
  currently an unwired comment. #metrics
- [ ] **Potential influence (empowerment proxy)** — outcome variance under wholesale
  preference redraws; complements exercised influence. `intervention_response`
  already covers influence-now. #metrics #alpha
- [ ] **Antifragility study v0** — regime shift via `ScheduleSpec.onset` (an AI
  growth-rate shock mid-run): does a defense calibrated for the old regime keep
  preserving influence? The harness expresses it; the study design is the work.
  #alpha
- [ ] **Schedule sweep** — vary quota cadence (k = 1 / 5 / 10) via `ScheduleSpec`
  triples in `experiments/benchmark/scenarios.py`. #alpha

## Later

### Observability epic, remaining rungs #observability

Sequenced so O0/O1 land before the connection; a live connection to an unversioned
store only moves the drift downstream.

- [ ] **O2: frontend to backend connection** — the load-bearing one. The playground
  embeds exporter-generated fixtures per game; replace with the frontend reading
  the O1 store, static versioned artifact dir first, API only if interactivity
  demands it. Keep the v1.x trajectory contract as the wire format
  (`docs/web-trajectory-contract.md`); this changes transport, not schema. This is
  also what retires the four hand-ported models in the web kernel (see the
  boundary section of `CLAUDE.md`). #public
- [ ] **O3: metric-kind to chart mapping** — one renderer per metric `kind` rather
  than per metric, so a new model's readouts get a correct chart with no bespoke
  frontend code. `system_graph`'s generic pipeline scene is the proof the pattern
  works.
- [ ] **O4: long-run payload budget** — decimation / level-of-detail in the
  trajectory contract; the wire payload is still O(T x N) for node fields. Pair a
  decimated series for display with the exact reduced scalars for numbers.
- [ ] **O5: streaming run writer** — emit scalars incrementally during a run;
  natural fit with the Reducer carry; gives crash-resumable partial results.
- [ ] **O6: decide the fate of the archived tkinter studio** — port its graph-editor
  idea onto the O2 web path or declare it superseded. Default: superseded.
  #decision

### Types epic: mechanisms as the centre of gravity #types

Framing decision (2026-08-07): the library's centre of gravity is mechanisms plus
type definitions; environments stay in-repo as reference substrates, the
conformance fixtures and evidence base for the mechanism catalog. Narrative split,
not repo split.

- [ ] **TYPE-01: slot datatype vocabulary** — the datatypes a mechanism can read or
  write (conserved ledger, distribution, network/adjacency, signal, plan), grounded
  in the ledger grammar of `docs/ledger-design.md`. Today `.reads` / `.writes` are
  bare field names, so a mechanism only ports to a substrate that spells its state
  the same way.
- [ ] **TYPE-02: key-parameterized mechanisms** — generalize the `sortition` pattern
  (`cfg.adj_key` in `mechanisms/political.py`) so mechanisms bind reads/writes to
  configurable slot names. Depends on TYPE-01.
- [ ] **TYPE-03: typed registry with targets** — each `REGISTRY` entry gets a
  signature card: input datatypes to output datatypes, target, family, provenance.
  Machine-readable, so the catalog answers "which mechanisms take a monetary ledger
  and return a distribution?". Depends on TYPE-01.
- [ ] **TYPE-04: strict compose mode** — `compile_pipeline` silently serializes
  write-write conflicts by design. Add a strict mode that raises on WAW between
  modules of different provenance; `_build_dependency_graph` already computes
  every WAW pair, only the policy is missing. Independent of TYPE-01.
- [ ] **BOUND-01: mechanisms and environments boundary** — mark the layering in
  `CLAUDE.md` (core + mechanisms + type vocabulary = stable layer; environments =
  reference substrates) and add an import-direction test: mechanism source never
  imports environments (already true; only mechanism tests do). #boundary

### Models and environments #environment

- [ ] **Couple `recovery` to remaining human-culture share** — removes the
  manufactured culture floor honestly (constant recovery holds share at about 0.59;
  zeroed, 0.008); the displacement corner then has a true absorbing state, which is
  the paper's actual claim.
- [ ] **AI listening drift** — relax the frozen AI rows in `influence_exchange`;
  they set the structural political floor, insensitive to every slider. See the
  open decision above.
- [ ] **Rescue probe for irreversibility** — run collapse, switch the full defense
  portfolio on mid-run via `ScheduleSpec.onset`, measure what recovers vs. what is
  locked in. Cheapest honest operationalization of "basin". #alpha
- [ ] **Consumption channel** — households supply labor and consume; currently
  unrepresentable in `compute_economy` (labor only).
- [ ] **State-dependence substrate** — tax base / coercion / legitimacy as its own
  model, not a coupling label; the political substrate currently measures voice
  only.
- [ ] **Communication mechanism (commons)** — cheap talk, the strongest measured
  lever in the CPR experimental literature (Ostrom, Walker and Gardner 1992); the
  commons card's strongest omission. #mechanism
- [ ] **Wire opinions to something or demote consensus** — opinions currently feed
  back into nothing; give them a downstream edge or label the readout decorative.
- [ ] **Learning delegate** — an `agents/` variant that adapts to reward, making
  sanction deterrence measurable (v0 sanctions only confiscate). Unblocked by the
  agent type contracts. #agent
- [ ] **Delegated voting dial** — route `vote` through the delegates' alignment
  blend in `governed_commons` (v0: principals vote directly).
- [ ] **Monitoring mechanism** — costly monitoring as its own catalog entry
  (promote from polycentric). #mechanism

### Catalog entries #mechanism

- [ ] **Democracy: representative** — fixed representatives, elected every T ticks.
- [ ] **Democracy: liquid** — revocable delegation every tick (Bloembergen 2019).
- [ ] **Market: double auction** — second variant, same type contract
  (Friedman 1993).
- [ ] **Market: sealed-bid** — third variant (Krishna 2002).
- [ ] **Network: broadcast** — share signal with neighbours; the network family's
  first entry (`mechanisms/network.py`). Distinct from `agents/broadcast.py`, which
  is a policy.
- [ ] **Network: trust-weighted** — credibility overlay on broadcast (Ostrom 1990).
- [ ] **Network: gossip** — probabilistic forwarding (Shah 2009).
- [ ] **Swap test** — replace posted-price with double auction, verify the same
  pipeline compiles.
- [ ] **`market.py` retrofit** — bring the pre-`@transform` market entry up to the
  catalog contract (declared reads/writes, no Python loops).

### Public and infrastructure

- [ ] **Website lab page sync** — flip the scenario-1 status chip, replace
  illustrative leaderboard rows with real scorecard JSON
  (`experiments/benchmark_commons/results/`). #public
- [ ] **Papers WP1 to WP3: referee pass and site links** — the drafts are complete
  and live outside the repo; an optional second referee or cold-read pass, then
  PDFs linked from the site. #public #paper
- [ ] **WSL2 + CUDA setup** — `jax[cuda12]` in WSL2, verify GPU. Every performance
  number so far is CPU. #infrastructure

## Done

- [x] **2026-09-30 Compile once per spec; compile cache for tests** — `EnvSpec.run`
  / `run_batch` cache one program per spec and call signature: repeat calls 3 to
  73 ms instead of 0.19 to 1.35 s, bit-identical to the eager scan in every
  registered environment. A plain `jax.jit` was not (three causes: compiled init,
  batched fields vmap leaves unbatched, closed-over arrays folded as constants),
  all handled in `EnvSpec._compiled` and held by `tests/test_spec_jit.py`;
  red-teamed. Root `conftest.py` turns on the persistent compile cache: full suite
  557 s to 200 s warm. Detail: `experiments/perf_baseline/README.md`. #performance
  #engine
- [x] **2026-09-11 O0 + O1: metric identity and the run store** — `@kind`
  declarations and transitive definition hashes on every catalog and environment
  metric (`cilib.metrics.identity`); one `cilib.runs` writer with a nested,
  hash-verified address block, used by the benchmark CLI, `examples/05`, and both
  fixture exporters; perf_baseline shares the engine block only. Kinds shipped as
  share / concentration / rate / level / index. Red-team record and limits in
  `docs/observability-design.md`; contract: `src/cilib/runs/README.md`.
  `assumption_audits` still writes nothing and stays on the card's list. #observability
- [x] **2026-09-10 Performance baseline** — `experiments/perf_baseline/`: compile
  costs two orders of magnitude more than running; the eager path recompiles every
  call; seeds are near free; dense ledgers grow at N^1.6 to N^1.8 past about 600
  nodes. #performance
- [x] **2026-09-10 Library explainer exporter** — `experiments/library_explainer/`:
  schema-validated fixtures for the site's explainer page, snippets extracted from
  engine source at the manifest's git rev. #public
- [x] **2026-09-10 GD game recorded dead** — both routes (precomputed branch tree,
  live levers behind a service); `docs/gd-game-postmortem.md`; the dead-end block
  in `CLAUDE.md`. Leftover code kept as record, not as a seed.
- [x] **2026-09-10 `ledger_society` v1** — per-tick substrate globals,
  `ai_insularity`, `reach_cut`, lever families; the transfer-gap sign flip recorded
  (`docs/ledger-design.md` section 9), not yet clean. #ledger
- [x] **2026-09-10 Docs** — router `CLAUDE.md` files, working ethos, `STANDARDS.md`
  v0, `docs/paper-style.md`; transcript voice stripped from public prose.
- [x] **2026-08-14 Web kernel parity audit** — four hand-ported models named and
  dated in the boundary section of `CLAUDE.md`; only the coupled model has measured
  parity. #public
- [x] **2026-07-30 GD suite v0.1** — `ledger_society` as the shared coupled model
  (`docs/gd-suite-v0.1.md`, `docs/ledger-design.md`); dial-lattice bundles;
  `webexport` as the single producer of the web payload. #ledger #public
- [x] **2026-07-30 WP1 to WP3 models and sweeps** — `capital_economy` (owned AI
  capital, capability growth law, typed resource manifest;
  `experiments/wp1_economy` E1 to E5: the end-state human share has no floor),
  `influence_exchange` instruments (`experiments/wp2_culture`), `delegative_polity`
  (`experiments/wp3_politics`). Paper drafts complete, kept outside the repo.
  #paper
- [x] **2026-07-28 Alpha A3 to A5 and the trajectory ceiling** — `value_contagion`
  (culture as contagion, four corners from two dials), `influence_exchange` plus
  the political family (`sortition`, `influence_cap`), `coupled_society` (three
  domains, one kappa dial, couplings neutral at kappa = 0 by construction); the
  Reducer tier (`core/reduce.py`: no metric across eight environments needs the
  joint (T, N) array); sparse adjacencies and the spectral C0 slice; `system_graph`
  and web contract v1.1; assumption audits and the cold read of the assumptions
  cards (`experiments/assumption_audits/`). #alpha
- [x] **2026-07-14 A2 `compute_economy`** — CES production ABM, fiscal mechanisms
  (`ai_revenue_tax`, `ownership_cap`), network generators, scheduler-attached
  mechanisms, multi-scenario harness. Finding: influence-from-birth about 1.0 in
  every regime; influence-now 0.34 undefended vs 0.79 to 0.83 defended.
  `docs/abm-suite-design.md`. #alpha
- [x] **2026-07-13 GameSpec boundary, A0, A1** — environments as open games with
  `close(game, policy)`; counterfactual influence by paired same-key rollouts
  (`docs/game-boundary-design.md`); `governed_commons` slice; benchmark harness
  with bootstrap CIs (`experiments/benchmark_commons/`). Finding: unenforced quota
  voting restores about zero causal influence; quota plus sanctions restores 0.91.
  #alpha
- [x] **2026-03 to 2026-04 Plans 1 to 3** — `@transform` reads/writes metadata, the
  pipeline compiler (topological order from declared sets, parallel batches),
  cycle and conflict validation, the schedule primitive, `parallel` and
  `conditional`, the `lax.scan` runner and `vmap` over seeds, the pytree partition
  of `global_attrs`, `GraphState`, posted-price market, direct democracy,
  concentration metrics. `docs/changelogs/2026-03-30_core-primitives.pdf`. #core
