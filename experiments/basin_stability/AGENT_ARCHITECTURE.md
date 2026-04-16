# Agent Architecture — Basin Stability Experiment

**Last updated:** 2026-04-16
**Status:** Working baseline. Mechanism differences visible around 40-70% adversarial.

This document explains exactly what each agent does, the game-theoretic reasoning behind each decision rule, and what has changed from the original Q-learning design.

---

## 1. The Big Shift: No Reinforcement Learning

**Original design (abandoned):** Independent tabular/linear Q-learning agents with epsilon-greedy exploration, TD(0) updates, reward-driven policy learning. Required 1000+ episodes to converge.

**Current design:** **Heuristic agents with adaptive trust dynamics.** Agents follow deterministic rules derived from game theory. The only adaptive state is a trust matrix updated via EMA.

### Why abandon RL?

1. **Burn-in cost**: RL agents need ~1000 episodes to converge. Basin stability measurement with 100 seeds × 3 mechanisms × 2 tracking modes × 7 adversarial fractions = 4,200 conditions. That's millions of episodes for convergence before measurement can even begin.
2. **Credit assignment is impossible**: All agents receive the same collective reward (resource delta). No individual signal differentiates a good voter from a bad one.
3. **The interesting policies are already known**: Given noisy signals, rational behavior is signal-greedy. RL would slowly rediscover what decision theory tells us directly.
4. **What's left to "learn" is trust**: Agents need to track WHO makes good decisions over time. That's not RL — it's Bayesian filtering / social learning.

The cooperative AI literature (Barfuss 2024, Lerer & Peysakhovich 2017) supports this: when the policy is analytically characterizable, parameterize it directly and study the dynamics. Use RL only for decisions you genuinely don't know how to make.

---

## 2. The Game Structure

**Environment**: Common-pool resource game. At each of T=200 rounds:
- 4 portfolios have yields `y_k ~ Beta(α_k, β_k) * (high_k - low_k) + low_k`
- Each agent `i` observes `ŷ_{i,k} = y_k + N(0, σ_i²)` — noisy private signal
- Signal quality `σ_i` is heterogeneous: 30% at σ=0.05, 40% at σ=0.20, 30% at σ=0.50
- Mechanism aggregates votes into a single selected portfolio `k*`
- Resource evolves multiplicatively: `R(t+1) = R(t) * y_{k*}(t)`
- System collapses if `R < 20` (R_min)

**Players**: N agents split into cooperative and adversarial types.
- Cooperative: wants high resource growth (preferences aligned with collective)
- Adversarial: wants collapse (preferences anti-aligned)

**Information structure**: All agents see the selected proposal, the realized yield, and everyone else's votes. This is what enables trust tracking.

---

## 3. The Per-Round Action Sequence

Here's what happens in a single round, for every agent:

### Step 1: Observe signals

Each agent `i` receives noise-perturbed signals `ŷ_{i,k}` for all K portfolios.

### Step 2: Compute top choice and confidence

```python
# In policies.py — approval_vote()
top_choice = argmax(signals) if cooperative else argmin(signals)
confidence = (sorted_signals[0] - sorted_signals[1]) / signal_range
```

- **Cooperative top_choice**: best-looking portfolio
- **Adversarial top_choice**: worst-looking portfolio
- **Confidence**: normalized gap between top-2 signals. Low confidence = top-2 look similar.

**Game theory**: This is rational under noisy signals. Given no other information, your best bet is your highest signal. Adversaries invert this because their payoff is inverted.

### Step 3: Decide whether to delegate (PLD only)

```python
# In policies.py — should_delegate()
delegate = confidence < snr_threshold * σ_i
```

- Agent delegates when their own confidence is low relative to their noise level
- **High-noise agents (σ=0.50)** have low confidence often → delegate often
- **Low-noise agents (σ=0.05)** have high confidence usually → almost never delegate
- **Adversaries never delegate** — enforced in transform via `is_delegating & ~is_adversarial`

**Game theory**: This is the **value-of-information calculation**. If your signal is too noisy to trust, defer to someone with better information (via trust scores). Tying delegation to σ_i is the natural way to make delegation a function of information quality — the very heterogeneity that motivates having LD in the first place.

### Step 4: If delegating, choose a target

```python
# In policies.py — delegate_target()
target = argmax(trust_scores_row)  # highest-trust agent
```

Delegators adopt their target's `top_choice` as their own vote. Delegation is non-transitive: if your target also delegates, the weight is lost.

### Step 5: Assign vote weight (mechanism-specific)

This is the **ONLY place mechanisms differ**. Everything else is shared.

| Mechanism | Weight rule |
|---|---|
| **PDD** (Direct) | `weight = 1.0` for everyone |
| **PRD** (Representative) | `weight = rep_mask` (1.0 for elected reps, 0.0 otherwise) |
| **PLD** (Liquid) | `weight[j] = (1 + delegation_count[j])` if `j` votes directly, else `0` |

The PLD rule comes straight from the paper's definition (Eq. 4): `w_j = 1 + |D_j|`.

### Step 6: Universal aggregation

```python
# In transforms.py — aggregation_transform
votes_onehot = one_hot(actions, K)            # (N, K)
vote_counts = sum(votes_onehot * weights, 0)  # (K,)
selected = argmax(vote_counts)                # single winner
```

**This is identical for all three mechanisms.** The only input that varies is `weights`. That's the compositional architecture — different weight-assignment transforms, same aggregation.

### Step 7: Resource update

`R(t+1) = R(t) * y_{k*}` where `k*` is the selected portfolio and `y_{k*}` is its realized yield (sampled from the portfolio's Beta distribution).

### Step 8: Trust update (the only "learning")

```python
# In policies.py — trust_update()
performance[j] = y_{top_choice[j]} / mean(y)
trust[i, j] = λ * trust[i, j] + (1 - λ) * performance[j]
```

Every agent `i` updates their trust in every other agent `j` based on how good `j`'s preferred portfolio turned out to be relative to the mean yield across all portfolios.

**This is a simple exponential filter (Bayesian EMA).** The key parameter:
- **λ = 0.9 (predictive)**: long memory. Trust reflects cumulative track record.
- **λ = 0.1 (non-predictive)**: short memory. Trust is dominated by the last 1-2 rounds.

Non-predictive mode ≠ no tracking. It's **recency-biased tracking** — the "captured by latest news" failure mode you describe in the paper.

### Step 9: Election update (PRD only, every E=10 rounds)

```python
# In policies.py — election_vote()
target_seats = ceil(n_reps / 2)  # majority control needed
ranked = argsort(trust_scores_row)

# Cooperative: target top-K most trusted (bloc voting)
# Adversarial: target bottom-K least trusted (bloc voting)
vote = ranked[N - 1 - (agent_idx % target_seats)]  # coop
vote = ranked[agent_idx % target_seats]            # adv
```

Each agent votes for candidate `(i % target_seats)` in their ranked preference list. This implements **optimal bloc voting** from committee election theory: to capture a committee of size `n_reps`, spread your votes evenly across the `⌈n_reps/2⌉` seats you need for majority control.

Top `n_reps` by vote count become the new representatives.

---

## 4. What is actually adaptive?

The system has exactly **one** adaptive component: the `trust_scores` matrix (N×N).

Everything else is:
- **Deterministic and derivable from first principles** (voting heuristic, delegation rule, bloc voting strategy)
- **Or just environment state** (resources, proposals, signals)

This means:
- **Convergence is immediate** — no burn-in, no training, no lost episodes
- **Results are interpretable** — you can trace every decision to a clear rule
- **The experimental variable is clean** — λ directly controls the adaptive dynamic

---

## 5. The Compositional Pipeline

Per your architecture diagram (Transforms.png), the step pipeline is built by composing atomic transforms. The mechanism determines WHICH transforms compose and in WHAT ORDER, but each atomic transform is identical.

```
PDD: proposal_gen >> voting >> equal_weight   >> aggregation >> resource >> reward >> trust >> step
PRD: proposal_gen >> voting >> rep_weight     >> aggregation >> resource >> reward >> trust >> election >> step
PLD: proposal_gen >> voting(sets weight)      >> aggregation >> resource >> reward >> trust >> step
                                                 ^^^^^^^^^^^
                                              single shared transform
```

- `aggregation_transform` is one function with zero mechanism branching
- `proposal_gen`, `resource`, `reward`, `trust`, `step` are all shared
- Mechanism-specific transforms: `equal_weight_transform`, `rep_weight_transform`, `make_election_transform`, and the delegation logic inside `make_voting_transform`

This is pure functional composition — `sequential(*transforms)` in `core/category.py`.

---

## 6. The Experimental Variables

The sweep is a 3×2×N factorial:

| Axis | Values | What it tests |
|---|---|---|
| Mechanism | PDD, PRD, PLD | How weight is allocated across voters |
| Tracking mode | predictive (λ=0.9), non-predictive (λ=0.1) | Quality of the information used for delegation/election |
| Adversarial fraction | 0% to 80% | Stress test — how much anti-alignment can the system absorb |

### Labeling convention

- **Predictive** mechanisms have access to long-term track records → PDD/PRD/PLD
- **Non-predictive** mechanisms operate on recency-biased information → DD/RD/LD

This maps to the paper's framing: the "Predictive" prefix signals that agents have access to performance history. Removing it means agents only see recent outcomes.

---

## 7. What the results should show (hypotheses)

1. **PDD vs DD**: No difference. Equal-weight voting doesn't use trust scores, so λ is irrelevant.
2. **PRD vs RD**: PRD should resist capture better because long-term trust prevents populist flips in elections. RD should show more adversarial capture at moderate adversary fractions.
3. **PLD vs LD**: Biggest effect. PLD concentrates weight on proven performers. LD delegates based on recent luck, which adversaries can game.
4. **Interaction effect**: The gap between predictive and non-predictive should be largest for PLD, because delegation is continuous (every round) whereas PRD elections are only every E=10 rounds.
5. **All mechanisms collapse above 50% adversaries**: once adversaries have majority, no aggregation rule can save cooperation. Delegation can't create votes that don't exist.

---

## 8. Key files and functions

| File | Contains |
|---|---|
| `policies.py` | `approval_vote`, `should_delegate`, `delegate_target`, `election_vote`, `trust_update` — all pure, vmappable |
| `state.py` | `create_initial_state` — defines pytree, portfolio params, agent attributes |
| `transforms.py` | Step pipeline, compositional (one transform per step phase) |
| `environment.py` | `BasinStabilityEnv`, `run_batched` — vmap harness |
| `run_experiment.py` | Sweep orchestration, CSV output |
| `plots.py` | 4-panel summary plot, trajectory plots |

---

## 9. Open questions (for future work)

- **Portfolio design**: Current 4 portfolios have 3 with E[y] > 1.0. Adversarial "bottom 2" approvals still include a growth portfolio. Consider rebalancing (e.g., 2 decline + 2 growth) to make adversarial capture more impactful.
- **n_reps scaling**: `n_reps` is currently fixed. For 100 agents, 15 reps (~15%) works. For general comparison, might scale `n_reps = ceil(0.15 * N)`.
- **Delegation chains**: Currently non-transitive (max 1 hop). Paper's full PLD allows chains — worth exploring as an ablation.
- **Deceptive adversaries**: Currently adversaries behave transparently (invert approvals, never delegate, bloc-vote in elections). A stronger model: adversaries that mimic cooperatives for first N rounds to build trust, then defect. This would test PLD's ability to handle strategic deception.
