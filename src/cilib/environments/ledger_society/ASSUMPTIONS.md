# ledger_society — assumptions card

**What this says a society is.** One population holding three conserved stocks —
money, attention, ballots — where every form of cross-domain influence is either
spending one stock to move another or a declared read of a collective variable.
The same attachment kernel reallocates attention and ballots; disempowerment, if
it occurs, is the three human shares draining toward whoever accumulates and
spends fastest.

**Assumptions.**

- Three ledgers with explicit contracts (`state.LEDGERS`): money is minted by
  production and leaves through consume/invest/broadcast/lobby (declared sinks —
  goods, machines, and the influence industry are outside the modeled loop);
  attention and ballots are pure per-row reallocation, floored by
  `self_weight_w` / `self_weight_d`. The ladder asserts all three every run.
- Three coupling channels, one dial each, all value-agnostic (no `node_types`
  test in any channel): `reach_per_spend` (money→attention, multiplicative
  bought reach, linear and unsaturated — a critic should press here),
  `attention_to_ballots` (culture→politics, a port read into delegation
  attractiveness), `regime_rate` (money→rules: lobby spend moves enforcement,
  direction set endogenously by the sign of each spender's net redistribution
  transfer). Each is exactly neutral at 0; the three dials are the knockout axes.
- **Institutional self-repair** (`repair_rate`, default 0.02): enforcement relaxes
  toward 1 at this rate — the polity's maintenance floor, the same native-reversion
  idiom as WP3's churn and value_contagion's recovery. Added 2026-07-30 after a
  probe showed the unrepaired channel is a pure ratchet (any positive `regime_rate`
  collapses enforcement; the dial only set the speed). With repair, equilibrium
  enforcement is set by the pressure/repair ratio and the regime dial has a real
  gradient (probed: 1.00/0.87/0.70/0.24/0.01 across rates 0–0.04). `repair_rate=0`
  restores the ratchet — the honest bracket, sweepable. The `regime_rate` default
  moved 0.10 → 0.01 (2026-07-30): 0.10 predates repair and sat beyond full
  collapse AND off the swept 0–0.04 lattice; 0.01 is mid-gradient (enforcement
  ≈ 0.70), a nothing-special point on the swept curve per the house rule.
- The economy leg is a REDUCED WP1, not the full `capital_economy`: two-factor
  production with an automation share `a = eK/(eK+H)`, no IO backbone, no upkeep
  settlement, consumption as a sink rather than a demand loop. The conservation
  discipline is kept; the circular flow is not. Forks wanting the full flow
  start from `capital_economy` (its card colocates there).
- Type-referencing that remains is substrate, not coupling, each instance
  deliberate: humans hold the labor slot; the tax payout goes to citizens
  (the franchise as an institution); AI kernel rows are frozen and AI belief is
  pinned (the reservoir idiom, inherited from influence_exchange); AI delegates
  carry `ai_ballot`. Asymmetry of behavior lives in default allocation
  *policies* (`human_alloc` / `ai_alloc`) — the agnosticism rungs swap them.
- The threat is a schedule plus a strategy: AI actors arrive with seed capital
  (title minted, declared) and allocate toward invest/broadcast/lobby. There is
  no persuasive-content advantage and no amplification multiplier — reach is
  only ever bought.
- Kernel exponents default to γ=1 (share-neutral organic attachment,
  Krapivsky–Redner) — the honest baseline; capability growth
  (`growth_rate`, saturating at `e_ceiling`) is the only compounding source.
- `entrenchment_gain` defaults 0: the WP3 lock-in channel exists but is off —
  any regime erosion at defaults is funded lobbying, not concentration.
- Absent: a demand loop (consumption sinks), a state actor with a budget,
  belief→economy coupling (v0's `converts_capitalize` has no analog yet),
  defense-mechanism wiring (`cap_scale` seams), churn in the attention kernel,
  and any schedule separation between domains (lockstep only in v1).
- **Intervention events** (added 2026-07-31 for the GD game,
  docs/gd-game-design.md; a deliberate dated exception to observability-first
  sequencing). Player-enacted defense cards enter two ways, both exactly
  neutral when unenacted (bit-identity rungs): the influence-cap card is a
  channel property (`reach_cut` scales `reach_per_spend` from
  `reach_cut_onset`; 0 = pre-card model, bit-identical), and the rest ride the
  mechanism slot as one composed transform
  (`mechanisms/interventions.py`) — AI wealth levy (conserving transfer),
  attention campaign (one-shot `alloc_pref` shift), fund-repair
  (wealth drip into the declared `intervention_spend` sink — the money
  ledger's fifth sink — buying bounded enforcement uplift), and enforcement
  debits (enacting a political card spends institutional capacity). Cards ARE
  type-referencing by design — they are player-directed institutions, not
  coupling channels, so the value-agnosticism rule does not bind them. Every
  price and rate is tuned-for-legibility, none anchored; the game's manifest
  repeats this typing wherever the numbers travel.
- **Five substrate parameters are now per-tick globals.** Added 2026-07-31 for
  the three-families design, docs/gd-game-three-families.md. `gamma_w`,
  `update_rate_w`, `churn`, `repair_rate` and `entrenchment_gain` were closed
  over by their transforms, so nothing could vary them during a run. Those five
  are exactly what the culture family shapes — the attention kernel's exponent
  and its speed — and what the politics family spends — the maintenance floor
  and the lock-in hazard — so a live lever needs them to be state, not config.
  Each now rides `global_attrs` as `<name>_now`, initialised from its config
  field in `state.py` and read per tick by `rewire_listening`,
  `rewire_delegation` and `update_regime`. This extends the existing
  `reach_cut_now` precedent to five more fields and no others. It is plumbing,
  not a model change: no default moved, no lever writes these globals yet, and
  the substitution is bit-identical to the closed-over pipeline. Checked at
  defaults and with all five set off-default, 4 seeds, 200 steps, every trace
  field, both adjacency ledgers and every final node and global array equal
  element-for-element. The modelling consequence is that these five stop being
  run constants once levers land: a claim quoting one of them then owes the
  reader the trajectory it took, not a single number.
- **The three channel dials can be per-tick globals, behind a flag.** Added
  2026-10-06 so `EnvSpec.run_grid` can sweep coupling with one compile.
  `traced_channels=True` seeds `reach_per_spend_now`, `attention_to_ballots_now`
  and `regime_rate_now`; off, the default, the channels are the closed-over
  constants as before. Unlike the five above, this lift is not bit-identical on
  the seed-batched paths: lifting `reach_per_spend` or `regime_rate` changed
  `run_batch` numbers by about 1e-7 at the defaults and by up to 0.003 in the
  per-tick human power share at maximum coupling, T = 400. Single rollouts did
  not change. The flag-off model was checked against the old code at T = 400 on
  every run path in five configs and is unchanged. A result made with the flag on
  says so and is compared only with results made with it on. Details in
  docs/run-grid-design.md, "Lifting can move numbers".

**Classical result reproduced.** Not one theorem — three contract identities on
every run: per-tick money conservation (Δtotal = mint − sinks), row-stochasticity
of both adjacency ledgers, ballot shares summing to 1; plus the sealing
identity — with the three dials at 0, an economy-dial change leaves
belief/attention/power trajectories BIT-IDENTICAL under the same key
(`tests/test_ladder.py`).

**The dial(s).** The three channel gains jointly; ladder facts at defaults
(seeds 4, T=300, committed as ordering only per house rules): coupled composite
< sealed composite; no-arrivals human income share > 0.9 and power share > 0.6;
humans buying reach gain attention share when the channel is isolated — at
defaults the same purchase is outspent 10–40× by AI budgets, which is the
dynamic under study, not an artifact.

**Instrument.** Sealed twins via the three dials (same key); per-dial sealing is
the edge-knockout design (docs/ledger-design.md §6) — the measured coupling
matrix and its ρ(A) are the follow-up, not yet built.

**Lineage.** The GD-suite product construction (docs/gd-suite-v0.1.md §4):
reduced `capital_economy` economy leg; `influence_exchange` attention/belief
idioms; `delegative_polity` ballot/vote/regime idioms with
`mechanisms/democracy.py` reused verbatim; both adjacency ledgers driven by
`environments/attachment.py` (the shared kernel's first single-home use).
Replaces `coupled_society`'s κ-modulations with spends and port reads; the
invariance run against that baseline is the exit criterion for headline claims.

**Status.** Live v1 (2026-07-30, this deposit): 8-rung ladder passing.
Not in the benchmark; not in suite v0.1 (candidate for its `coupled` slot at
v0.2 after the invariance run). Referee paragraph: the linear unsaturated reach
channel makes late-run AI attention dominance partly a functional-form choice;
the consumption sink severs WP1's demand-attribution logic (an economy that no
longer needs human spending cannot be detected here); and the flywheel ordering
at defaults rests on hand-set channel gains — sign only, no magnitudes, until
the swept phase maps and the coupled_society invariance run exist.
