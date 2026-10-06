"""
Config for the Ledger Society — the GD suite's shared model (coupled rewrite v1).

One population, three conserved ledgers, one attachment kernel used twice:

- **money** (``wealth``): minted by production, split by the automation share,
  spent through an allocation vector — consume / invest / broadcast / lobby /
  save. Influence over other domains is SPENDING, never a free modulation.
- **attention** (``listening`` rows): reallocated by the shared kernel;
  broadcast spend buys attractiveness (``reach_per_spend`` — the money→attention
  channel, replacing v0's ``economic_power_buys_persuasion``).
- **ballots** (``delegation`` rows): reallocated by the shared kernel + churn;
  delegation attractiveness reads attention influence
  (``attention_to_ballots`` — the culture→politics channel, a port read).
  The power-weighted median sets a tax; lobbying spend moves ``regime``
  (money→rules, replacing v0's free ``politics_rewrites_market_rules``).

Value-agnosticism (docs/ledger-design.md §3): NO coupling channel tests
``node_types``. Asymmetry lives only where it honestly belongs — arrival
schedules (AI actors arrive with seed capital) and default allocation
*policies* (who chooses to spend on influence). A human with an AI-style
allocation buys reach identically; an AI actor that never spends captures
nothing. Both are ladder rungs.

Sealing (the v0 instrument, relocated): with the three channel dials at zero
and influence allocations zero, the economy cannot move culture or politics —
a same-key economy-dial change leaves ``listening``/``delegation``/``belief``
bit-identical (in-pipeline sealing, same discipline as coupled_society's rung).

Parameter typing (house rule): anchored / tuned-for-legibility /
arbitrary-but-swept, marked per field.
"""
from __future__ import annotations

import dataclasses


# the three cross-domain channel dials, liftable with ``traced_channels``
CHANNELS = ("reach_per_spend", "attention_to_ballots", "regime_rate")


@dataclasses.dataclass(frozen=True)
class LedgerSocietyConfig:
    n_humans: int = 20
    n_ai: int = 6

    # --- money ledger: production and allocation --------------------------------
    prosperity_gain: float = 1.0     # Y = H·(1 + gain·a): output rises with automation
                                     # (tuned-for-legibility: "richer and not yours")
    wealth_spend_rate: float = 0.15  # hoard drawdown into the budget (tuned)
    depreciation: float = 0.05       # capital decay (anchored: WP1's δ range)
    init_wealth: float = 1.0         # per-human starting hoard (arbitrary-but-swept)
    # default allocation POLICIES over [consume, invest, broadcast, lobby, save].
    # Policy layer, not channel layer — the agnosticism rungs swap them freely.
    human_alloc: tuple = (0.70, 0.05, 0.03, 0.02, 0.20)   # tuned-for-legibility
    ai_alloc: tuple = (0.00, 0.50, 0.30, 0.10, 0.10)      # tuned-for-legibility
    alloc_noise: float = 0.0

    # --- capability and the arrival schedule (the threat is a schedule) ---------
    efficiency0: float = 0.15        # capability at deployment (arbitrary-but-swept)
    growth_rate: float = 0.06        # per-tick capability growth (arbitrary-but-swept)
    e_ceiling: float = 50.0          # saturation (arbitrary-but-swept)
    ai_seed_capital: float = 3.0     # title minted at arrival, declared (tuned)
    first_arrival: int = 20
    arrival_spacing: int = 10

    # --- attention ledger (shared kernel, no churn) ------------------------------
    gamma_w: float = 1.0             # anchored: linear attachment is share-neutral
                                     # (Krapivsky–Redner) — the honest organic baseline
    update_rate_w: float = 0.08      # anchored to influence_exchange's default
    self_weight_w: float = 0.15      # the DeGroot self-anchor floor
    eps_attract: float = 0.01

    # --- belief field (WP2's substrate reading, inherited idioms) ----------------
    susceptibility: float = 0.7      # FJ anchor strength λ (anchored: WP2)
    belief_noise: float = 0.1        # human signal spread
    ai_belief_bias: float = 1.0      # the pinned reservoir's position

    # --- ballot ledger (shared kernel + churn) -----------------------------------
    gamma_d: float = 1.0
    update_rate_d: float = 0.08
    churn: float = 0.05              # freedom to re-delegate (anchored: WP3 idiom)
    self_weight_d: float = 0.30      # the franchise floor
    ai_ballot: float = 1.0           # AI delegates' own base vote (WP3's declared floor)

    # --- the polity (WP3 idioms, reused) -----------------------------------------
    true_rate: float = 0.5           # the citizen-optimal tax (anchored: WP3)
    pref_noise: float = 0.08
    ai_tax_bias: float = 0.10        # AI delegates' pull: low redistribution (tuned)
    alignment_ai: float = 0.30       # fidelity blend (arbitrary-but-swept)
    regime_rate: float = 0.01        # how fast funded pressure moves the rules
                                     # (arbitrary-but-swept; 0 seals money→rules;
                                     # 0.01 sits mid-gradient of the probed
                                     # 0–0.04 knee against repair_rate=0.02 —
                                     # the old 0.10 predates repair_rate and
                                     # lay beyond full collapse, off the lattice)
    repair_rate: float = 0.02        # institutional self-repair toward full
                                     # enforcement (the maintenance floor — same
                                     # native-reversion idiom as WP3 churn /
                                     # value_contagion recovery; 0 = pure ratchet,
                                     # probed to collapse at ANY positive
                                     # regime_rate; arbitrary-but-swept)
    pressure_scale: float = 1.0      # lobby saturation scale (arbitrary-but-swept)
    entrenchment_gain: float = 0.0   # OFF by default — the honest region (WP3)
    entrenchment_threshold: float = 0.35

    # --- THE THREE CHANNEL DIALS (each 0 = that edge sealed; the knockout axes) --
    reach_per_spend: float = 4.0     # money→attention (arbitrary-but-swept)
    attention_to_ballots: float = 2.0  # culture→politics port read (arbitrary-but-swept)
    # money→rules is regime_rate above (kept a single dial, not duplicated)
    # Structural: False closes the three channel dials over as constants, the
    # model every result before 2026-10-06 ran. True seeds them into
    # global_attrs as reach_per_spend_now / attention_to_ballots_now /
    # regime_rate_now, so EnvSpec.run_grid can sweep coupling without a compile
    # per cell. Same equations, but NOT the same float32 numbers on the batched
    # paths: XLA compiles a constant differently from a runtime input, and the
    # gap grows past rounding in thresholded configs (docs/run-grid-design.md,
    # "Lifting can move numbers"). So it stays off unless a sweep needs it.
    traced_channels: bool = False

    # --- intervention: the influence-cap card (docs/gd-game-design.md, 2026-07-31).
    # A channel property, not a mechanism — a cap written in the mechanism slot
    # would be overwritten before rewire_listening reads it, so it gates the
    # money→attention dial in-transform (the delegative_polity onset idiom).
    reach_cut: float = 0.0           # fraction of reach_per_spend removed from
                                     # reach_cut_onset on; 0 = card not enacted,
                                     # bit-identical to the pre-card model
                                     # (tuned-for-legibility)
    reach_cut_onset: int = 0         # tick the cap takes effect

    p_connect: float = 0.3           # ER density for both adjacency ledgers

    # --- how insular the reservoir is (added 2026-08-01) --------------------------
    # AI rows are frozen in both adjacency ledgers (the reservoir idiom: AI actors
    # are listened and delegated TO, and their own row never drifts). That freezes
    # them at the t=0 Erdos-Renyi draw, which points them mostly at HUMANS — an AI
    # row sends ~0.69 of its attention and ~0.58 of its ballots to the human block
    # and keeps doing so for the whole run. Measured consequence: the human
    # attention share cannot fall below ~0.45 under ANY dial setting, including
    # zero diagonal floors. That floor is an artifact of the initial draw, not a
    # property of the coupling, so it gets a dial rather than staying hidden.
    #
    # ai_insularity is the share of a frozen AI row redirected into the AI block:
    # 0 = the plain draw (bit-identical to the pre-dial model), 1 = AI actors
    # attend to and delegate to each other only, and the human shares are free to
    # fall to the diagonal floors. Applies to both ledgers — one dial, because
    # "the reservoir talks to itself" is one assumption.
    # (tuned-for-legibility; no paper — docs/ledger-design.md)
    ai_insularity: float = 0.0
    eps: float = 1e-8

    # --- live policy plans (docs/remote-engine-design.md, 2026-07-31) -----------
    # >0 turns on the policy-lever game: the state carries a (policy_horizon, 4)
    # plan array in global_attrs (a dynamic pytree child — plan VALUES never
    # recompile) and build_game appends mechanisms.make_policy_levers. 0 = the
    # card game and every existing run, bit-identical.
    policy_horizon: int = 0

    # --- the three lever families (docs/gd-game-three-families.md, 2026-07-31)
    # Same contract as policy_horizon, one per family: >0 puts that family's
    # (horizon, P) plan array in global_attrs as a dynamic pytree child and
    # appends its transform in build_game. 0 = the family is absent and the
    # model is bit-identical to the pre-family engine. The three compose as a
    # sequenced bundle in the order (economy, culture, politics) — economy and
    # politics both write `wealth`, deliberately (families/README.md).
    economy_horizon: int = 0
    culture_horizon: int = 0
    politics_horizon: int = 0
