# Mycorrhiza Development Context

**Last Updated:** January 22, 2026
**Session:** Agent System Development + JAX Parallelization

## What We Built

### 1. Core Agent System (`core/agents.py`)

Pure functional agent interface for GraphState transformations:

```python
# Minimal interface
class Policy(Protocol):
    def __call__(self, obs: ObservationMatrix, key: PRNGKey) -> ActionMatrix: ...

# Core functions
get_observation(state: GraphState, agent_id: int) → ObservationMatrix
apply_action(state: GraphState, agent_id: int, action: ActionMatrix) → GraphState
```

**Design Principles:**
- Pure functions only (no classes with state)
- JAX-native (works with vmap, jit, grad)
- Generic (no hardcoded game mechanics)
- Minimal (SOLID principles)

### 2. Concrete Policies (`agents/`)

**Simple Policies:**
- `RandomPolicy` - Random actions
- `TitForTatPolicy` - Reciprocal cooperation

**Learnable Policies:**
- `LinearPolicy` - Gradient-trainable (W @ obs + b)

All follow signature: `(obs: Array, key: PRNGKey) → action: Array`

### 3. Central Lake Model (`simple_lake_test.py`)

**Architecture:**
- Central lake node with fish population
- Agents extract from lake
- Fish regenerate with logistic growth (8% per round)
- Two agent types:
  - Sustainable: Extract 5 fish/round (fixed)
  - Exploiter: Extract 15% of lake (scales with population)

**Key Results:**
| Exploiter % | Collapse Round | Interpretation |
|-------------|----------------|----------------|
| 0%          | Round 51       | Slow decline   |
| 20%         | Round 6        | Rapid collapse |
| 50%         | Round 2        | Very rapid     |
| 70%+        | Round 0        | Immediate      |

**Why it works:**
- Initially: 80 fish regenerate/round, 50 extracted (sustainable)
- Population drops → regeneration drops
- Eventually: extraction > regeneration → collapse
- Exploiters accelerate by extracting 150 fish vs 5 fish

**Metrics Tracked:**
- Lake fish over time
- Total system fish (lake + agents)
- Agent resources (cumulative extraction)
- Per-round extraction (average & max)

### 4. Democratic Fishing System (`democratic_fishing_test.py`)

**Components:**
- 3 fish species (Herring, Salmon, Cod)
- 4 extraction portfolios (Conservative, Balanced, Aggressive, Species-focused)
- Prediction markets (noisy signals about portfolio outcomes)
- Three democratic mechanisms:
  - **PDD** (Predictive Direct Democracy): All vote equally
  - **PRD** (Predictive Representative Democracy): Elected representatives
  - **PLD** (Predictive Liquid Democracy): Performance-based delegation
- Aligned and adversarial agents
- Messaging between agents

**Current Status:**
- All mechanisms implemented
- Agents vote on portfolios
- Fish regenerate and populations track
- Results: Systems survive even with 50% adversaries (need tuning)

### 5. JAX Parallelization Test (`test_jax_parallelization.py`)

**Setup:**
- Sequential baseline (loop over simulations)
- Vectorized with vmap (parallel simulations)
- JIT compilation for both
- Tests batch sizes: 2, 4, 8, 16, 32
- Verifies results match exactly

**CPU Results:**
- Average speedup: **1.3x**
- Batch 4: 1.21x
- Batch 8: 1.34x
- Batch 16: 1.27x
- Batch 32: 1.43x

**Why only 1.3x on CPU?**
- This is expected for CPU parallelization
- Already JIT-compiled and efficient
- Memory bandwidth limited
- Small batch sizes

**GPU Expectations:**
- Same code on GPU: **10-100x speedup**
- Zero code changes needed
- Just install JAX with CUDA in WSL

**Architecture Validation:**
- ✅ Pure functional simulation works with vmap
- ✅ JIT compilation successful
- ✅ Results deterministic and verifiable
- ✅ Ready for GPU deployment

## Key Learnings

### 1. Action Semantics Matter

**Three models tested:**

**Transfer semantics (first test):**
- Actions = "send resources to other agents"
- Result: Zero-sum + regeneration = exponential growth
- Not a tragedy, just redistribution with growth

**Independent extraction (v2 test):**
- Each agent extracts from all others
- Result: 5 agents × 10% = 50% total extraction
- Immediate collapse (too aggressive)

**Central hub extraction (final):**
- Agents extract from shared lake
- Result: Clear collapse gradient
- ✅ This models tragedy of commons correctly

### 2. Architecture is Solid

**What works:**
- GraphState holds all state immutably
- Observations extract agent-local views
- Policies map obs → action cleanly
- Actions update state functionally
- JAX transformations work throughout

**What's flexible:**
- Easy to swap action semantics
- Same GraphState for different games
- Policies are composable
- No framework changes needed for new mechanics

### 3. JAX Integration Success

**Parallelization:**
- vmap works for batch simulations
- 1.3x on CPU, expected 10-100x on GPU
- Factory pattern for static shapes works
- jax.lax.scan for efficient loops

**Key patterns:**
```python
# Factory for static shapes
def make_run_simulation(n_rounds: int):
    @jit
    def run(agent_types, initial_state, key):
        # ... uses n_rounds as static value
        return final_state
    return run

# Vectorize over keys (batch dimension)
run_vec = jit(vmap(run_sim, in_axes=(None, None, 0)))
```

## Thesis Context

### Research Questions

From "Red Teaming Democracy" (Jonas Hallgren, Uppsala, 2026):

**RQ1:** How does PLD compare to PDD and PRD under adversarial pressure?
**RQ2:** What adversarial concentration causes system breakdown?
**RQ3:** How to develop effective red team agents?
**RQ4:** What role does cognitive framing play in red team effectiveness?

### Key Thesis Findings

1. **PLD showed performance advantages** under moderate adversarial pressure (25-50%)
   - Adaptive delegation filters out poor performers
   - Breakdown thresholds: 70-90% adversaries (higher than expected)

2. **Red team sub-optimality** was the key challenge
   - LLM agents struggled to maintain destructive behavior
   - Effectiveness highly dependent on cognitive framing
   - "Competitive market pressure" framing most effective

3. **Iterative validation protocol** was necessary
   - Can't assume LLMs will execute goals faithfully
   - Need baseline tests with perfect information
   - Debug in simplified environments first

### This Codebase's Role

Provides the **functional JAX-native substrate** for thesis research:
- Pure functional transformations (not OOP)
- Parallelizable with vmap (1000+ simulations)
- Deterministic and reproducible
- Fast (JIT compiled, GPU-ready)

## Technical Decisions

### Why Pure Functional?

**Benefits:**
- Easy to reason about (no hidden state)
- Easy to parallelize (no mutation conflicts)
- Easy to test (deterministic)
- Easy to compose (transformations chain)

**Trade-offs:**
- More verbose than OOP
- Requires discipline (no cheating with mutation)
- Learning curve for imperative programmers

### Why JAX?

**Benefits:**
- vmap for automatic parallelization
- jit for speed without manual optimization
- grad for learning (future work)
- GPU support with zero code changes

**Trade-offs:**
- Functional purity required
- Static shapes needed for JIT
- Windows GPU requires WSL

### Why GraphState?

**Benefits:**
- General enough for any agent system
- Structured enough to be useful
- Immutable by design
- Works with JAX transformations

**Trade-offs:**
- Dict-based interface less type-safe than dataclasses
- Some overhead from immutable updates
- Learning curve for graph thinking

## File Organization

```
Mycorrhiza/
├── core/
│   ├── graph.py           # GraphState definition
│   ├── agents.py          # Agent interface (NEW)
│   ├── initialization.py  # State initialization (NEW)
│   ├── category.py        # Transform composition
│   └── property.py        # Invariants
│
├── agents/                # Concrete policies (NEW)
│   ├── simple.py          # RandomPolicy, TitForTatPolicy
│   └── learnable.py       # LinearPolicy
│
├── engine/
│   └── environments/
│       └── resource_game.py  # Resource game initialization
│
├── plans/                 # Documentation (NEW)
│   ├── CONTEXT.md         # This file
│   ├── ARCHITECTURE.md    # System design
│   └── NEXT_STEPS.md      # Future work
│
├── simple_lake_test.py           # Working lake model (NEW)
├── democratic_fishing_test.py    # Democratic mechanisms (NEW)
├── test_jax_parallelization.py   # GPU benchmark (NEW)
│
└── [legacy test files]
    ├── quick_test_fishing.py
    ├── quick_test_fishing_v2.py
    └── TEST_RESULTS.md
```

## Current Status

### ✅ Completed
- Core agent system (pure functional)
- Central lake model (working, validated)
- Democratic mechanisms (implemented)
- JAX parallelization (CPU validated)
- Concrete policies (simple + learnable)

### ⏳ In Progress
- GPU benchmarking in WSL
- Democratic mechanism comparison under stress
- End-to-end resilience tests

### 📋 Next Steps
1. GPU benchmark in WSL (verify 10-100x speedup)
2. Tune democratic fishing for visible mechanism differences
3. Run parallel sweeps over adversarial proportions
4. Implement learned policies with gradient descent
5. Visualize population trajectories

## How to Continue

### For GPU Testing
```bash
# In WSL
cd /mnt/c/Users/Jonas/Documents/GitHub/Mycorrhiza
python3 test_jax_parallelization.py

# Should see [cuda(id=0)] and 10-100x speedup
```

### For Democratic Mechanism Testing
```bash
# Tune parameters in democratic_fishing_test.py
# Make extraction more aggressive or regeneration slower
# Run multiple trials with different adversarial proportions
python democratic_fishing_test.py
```

### For Learning Experiments
```python
# Use LinearPolicy with optax for gradient descent
import optax

policy = LinearPolicy(input_dim=obs_dim, output_shape=action_shape)
optimizer = optax.adam(learning_rate=0.01)

# Train loop:
# 1. Get observation
# 2. Policy forward pass (with params)
# 3. Compute loss (resources gained/lost)
# 4. Gradient step with optax
# 5. Update params
```

## References

### JAX Resources
- [JAX vmap Performance](https://apxml.com/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/vmap-performance)
- [Parallel Training JAX](https://willwhitney.com/parallel-training-jax.html)
- [JAX Documentation](https://jax.readthedocs.io/)

### Theoretical Background
- Ostrom, E. (1990). Governing the Commons
- Berinsky et al. (2025). Tracking Truth with Liquid Democracy
- Hagberg & Kazen (2024). Predictive Liquid Democracy
- Hallgren, J. (2026). Red Teaming Democracy (thesis)

### Implementation Patterns
- Park et al. (2023). Generative Agents (LLM simulation)
- Vezhnevets et al. (2023). Concordia (agent framework)

---

**For Questions:** Check ARCHITECTURE.md for system design, NEXT_STEPS.md for roadmap.
