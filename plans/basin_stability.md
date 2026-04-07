# Plan: Basin Stability Experiment — Implementation & Runtime Estimate

## Context

Paper: "Basin Stability of Democratic Mechanisms Under Adversarial Pressure". Linear Q-learning agents in a proposal-selection resource game, comparing PDD/PRD/PLD under adversarial sweeps, measured via basin stability (500 seeds × 7 adversarial levels × 3 mechanisms).

The `experiments/governed_harvest/` directory has a working experiment with transforms, environment, and runner — but it implements a **different game** (logistic regrowth + harvest extraction with bandit agents). The paper requires a multiplicative proposal-selection game with linear Q-learning agents.

**The infrastructure is solid**: GraphState pytree partitioning is done, `jax.vmap` works in transforms, the transform pipeline architecture works, the environment ABC works. What needs changing is the **game model** and **agent model**.

**Adversarial agents**: Same Q-learners with inverted reward (`r = -(R(t+1) - R(t))`). Both agent types share identical architecture; only reward sign differs. Dispatched via `jnp.where(is_adversarial, -reward, reward)` in the reward transform.

---

## Gap Analysis: What Exists vs What's Needed

| Component | Current (governed_harvest) | Paper spec | Gap |
|-----------|--------------------------|------------|-----|
| **Agent model** | Bandit (softmax over weights) | Linear Q-learning: Q(s,a) = w_a^T s + b_a, TD(0) | **Rewrite** |
| **Resource dynamics** | Logistic regrowth - harvest extraction | Multiplicative: R(t+1) = R(t) × u_{k*} | **Rewrite** |
| **Decision structure** | Choose harvest level (continuous) | Choose proposal k ∈ {1,...,K} (discrete) | **Rewrite** |
| **Information** | All agents see same resource level | Heterogeneous noisy signals σ∈{0.05,0.20,0.50} | **New** |
| **PDD** | Median of vote values | Plurality of proposal votes | **Modify** |
| **PRD** | Mean of rep vote values, fixed reps | R=3 reps, elections every E=10 rounds | **Modify** |
| **PLD** | Brier-weighted votes | Delegation graph + EMA trust + weighted plurality | **Rewrite** |
| **Metrics** | Survival time, final resources | Basin stability, delegation Gini, capture rate | **New** |
| **Sweep** | 4 mechanisms × 4 fractions × 30 reps | 3 mechanisms × 7 fractions × 500 seeds + baselines | **Modify** |
| **Transform pipeline** | vote→harvest→resource→reward→learning→prediction→step | vote→aggregate→resource_update→reward→q_update→trust_update→step | **Restructure** |
| **Pytree partitioning** | Done ✓ | — | — |
| **vmap in transforms** | Done ✓ (agent-level) | Need episode-level vmap too | **Add** |

---

## Implementation Plan

### Approach: New experiment directory, reuse core infrastructure

Create `experiments/basin_stability/` alongside the existing `governed_harvest/` (don't modify it — it may be needed for comparison). Reuse `core/graph.py`, `core/environment.py`, `core/simulation.py`, `core/category.py`, `core/pipeline.py`.

### Files to create

**`experiments/basin_stability/__init__.py`** — empty

**`experiments/basin_stability/state.py`** (~80 lines)
- `create_initial_state(n_agents=20, n_adversarial, K=4, T=200, seed, ...)`
- Node attrs: `q_weights` (N × K+1 × K+2 for PLD, N × K × K+1 for PDD/PRD), `q_bias`, `trust_scores` (N × N), `signal_quality` (N,)
- Global attrs: `resource_level`, `proposals` (K,), `signals` (N,K), `step`, `rng_key`, static config
- Signal quality assignment: 30% σ=0.05, 40% σ=0.20, 30% σ=0.50

**`experiments/basin_stability/policies.py`** (~100 lines)
- `q_select_action(q_weights, q_bias, state_vec, key, epsilon)` — ε-greedy over Q(s,a) = w_a^T s + b_a
- `q_update(q_weights, q_bias, state_vec, action, reward, next_state_vec, alpha, gamma)` — TD(0) weight update
- `trust_update(trust_scores, voted_proposal, actual_utility, mean_utility, lambda_)` — EMA update
- `delegate_target(trust_scores, is_adversarial)` — argmax trust (or argmin if adversarial)

**`experiments/basin_stability/transforms.py`** (~200 lines)
- `proposal_generation_transform` — draw K utilities from U(0.80, 1.25), generate noisy signals per agent
- `voting_transform(mechanism)` — Q-learning action selection, vmapped over agents
- `aggregation_transform(mechanism)` — PDD plurality / PRD rep-only plurality / PLD weighted plurality
- `resource_update_transform` — R(t+1) = R(t) × u_{k*}
- `reward_transform` — r = R(t+1) - R(t) for aligned, -(R(t+1) - R(t)) for adversarial
- `q_learning_transform` — TD(0) update on Q-weights, vmapped over agents
- `trust_update_transform` — EMA performance tracking
- `election_transform` — PRD: every E rounds, elect R=3 reps by trust-based voting
- `step_counter_transform`

**`experiments/basin_stability/environment.py`** (~60 lines)
- `BasinStabilityEnv(mechanism, n_agents, n_adversarial, seed, K=4, T=200, ...)`
- Step transform = `sequential(proposal_gen, voting, aggregation, resource_update, reward, q_learning, trust_update, election, step_counter)`

**`experiments/basin_stability/run_experiment.py`** (~120 lines)
- `run_sweep(mechanisms, adversarial_fractions, n_seeds=500, ...)`
- `compute_basin_stability(trajectories, R_coop, eval_start=150)`
- `run_baselines(mechanism, p_adv, n_seeds)` — random, oracle, heuristic agents
- JSON output with Wilson CIs
- Optional: vmap over seeds for acceleration

**`experiments/basin_stability/analysis.py`** (~80 lines)
- `basin_stability_curve(results)` — BS ± Wilson CI per mechanism
- `breakdown_threshold(results)` — min p where BS < 0.5
- `delegation_gini(trajectories)` — PLD weight concentration
- `capture_rate(trajectories)` — PRD adversarial rep fraction
- `fisher_exact_pairwise(results)` — Holm-Bonferroni corrected comparisons

---

## Paper Spec Quick Reference

### Environment
- N=20 agents, K=4 proposals/round, T=200 rounds
- Proposals: u_k ~ Uniform(0.80, 1.25) regenerated each round
- Resources: R(t+1) = R(t) × u_{k*}, collapse if R < 20
- Signal quality: σ ∈ {0.05, 0.20, 0.50} assigned 30%/40%/30%
- Noisy signals: û_{i,k} = u_k + ε, ε ~ N(0, σ_i²)

### Agent Architecture
- State: s_i = [R(t), û_{i,1}, ..., û_{i,K}] ∈ ℝ^5
- Q-function: Q(s, a) = w_a^T s + b_a (30 params per agent for PLD)
- Update: TD(0) with α=0.01, γ=0.95
- Exploration: ε decays 1.0→0.05 over first 100 rounds
- Trust: EMA with λ=0.9, ρ_j = u_{a_j} / mean(u_k)

### Mechanisms
- PDD: equal-weight plurality voting
- PRD: R=3 elected reps, elections every E=10 rounds, reps-only plurality
- PLD: vote or delegate action, delegation to argmax trust, weighted plurality

### Rewards
- Aligned: r = R(t+1) - R(t)
- Adversarial: r = -(R(t+1) - R(t))

### Sweep
- p_adv ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
- 500 seeds per condition
- Train: rounds 1-150, Evaluate: rounds 151-200
- Basin stability: fraction of seeds where mean eval resources > R_coop

### Baselines
- Random: uniform action selection
- Oracle: perfect information, always best proposal
- Heuristic: vote for highest signal (no learning, no delegation)

---

## Computational Budget

| Mode | Total time |
|------|-----------|
| Python loop, CPU | ~3-5 hours |
| JIT + vmap over seeds | ~5-10 min |
| JIT + vmap + GPU | ~1-3 min |

Total: 52,500 runs (10,500 main + 31,500 baselines + 10,500 ablation)

---

## Verification Plan

1. **Unit test Q-learning**: single agent, fixed proposals, verify Q-values converge to select best proposal
2. **Mechanism correctness**: 
   - PDD: verify plurality winner matches manual count
   - PRD: verify only reps vote, elections rotate every E rounds
   - PLD: verify delegation flows to highest-trust agent, adversarial weight decreases
3. **Baseline validation**:
   - Oracle agents + 0 adversaries → BS ≈ 1.0
   - Random agents → BS << 1.0 (confirms game is non-trivial)
4. **Convergence check**: plot action entropy over rounds, verify stabilization before round 150
5. **Full sweep**: run with n_seeds=10 first (quick), then scale to 500

---

## Key Files to Reuse

| File | What to reuse |
|------|--------------|
| `core/graph.py` | GraphState with pytree partitioning (DONE) |
| `core/environment.py` | Environment ABC with transform-based step |
| `core/category.py` | `sequential()` for composing transforms |
| `core/simulation.py` | `Simulation` class with run loop |
| `experiments/governed_harvest/state.py` | Pattern for create_initial_state |
| `experiments/governed_harvest/transforms.py` | Pattern for vmap-compatible transforms |
| `experiments/governed_harvest/policies.py` | Pattern for pure policy functions |
