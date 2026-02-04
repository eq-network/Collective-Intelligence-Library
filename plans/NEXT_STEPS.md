# Next Steps & Roadmap

## Immediate (This Week)

### 1. GPU Benchmarking ⏳
**Goal:** Verify 10-100x speedup on RTX 3050

**Steps:**
```bash
# In WSL
cd /mnt/c/Users/Jonas/Documents/GitHub/Mycorrhiza
python3 test_jax_parallelization.py
```

**Expected:**
- Device shows `[cuda(id=0)]`
- Speedup: 10-100x for batch sizes 32-256
- Memory usage fits in 4GB VRAM

**Success Criteria:**
- ✅ GPU detected by JAX
- ✅ Speedup > 10x vs CPU
- ✅ Results match CPU baseline
- ✅ Benchmark completes without OOM

### 2. Democratic Mechanism Stress Test 📋
**Goal:** See clear differences between PDD, PRD, PLD under adversarial pressure

**Current Issue:** All mechanisms survive at 50% adversaries

**Fix Options:**
1. **Make extraction more aggressive**
   - Increase exploiter rates (15% → 30%)
   - Reduce regeneration (8% → 4%)

2. **Add more fish species with dependencies**
   - Cod depends on Herring population
   - Salmon depends on both
   - Makes choices more consequential

3. **Make adversaries smarter**
   - Vote for worst portfolios consistently
   - Coordinate attacks on representatives (PRD)
   - Try to become delegates then defect (PLD)

**Success Criteria:**
- ✅ Clear separation between mechanisms
- ✅ PLD shows resilience advantage at 40-60% adversaries
- ✅ Breakdown thresholds differ by mechanism

### 3. Documentation Cleanup 📝
**Goal:** Consolidate test results

**Action:**
- ✅ CONTEXT.md created
- ✅ ARCHITECTURE.md created
- ✅ NEXT_STEPS.md (this file)
- 📋 Update README.md with link to plans/
- 📋 Archive old test files (quick_test_*.py)

## Short Term (Next 2 Weeks)

### 4. Parallel Adversarial Sweeps
**Goal:** Replicate thesis methodology with JAX parallelization

**Implementation:**
```python
# Define sweep parameters
adversarial_proportions = jnp.linspace(0.0, 0.9, 10)
mechanisms = ["PDD", "PRD", "PLD"]
n_replications = 100

# Vectorize over all combinations
def run_single_config(adv_prop, mechanism, key):
    return run_simulation(adv_prop, mechanism, key)

# vmap over batch dimension
run_batch = vmap(run_single_config, in_axes=(0, 0, 0))

# Run all configurations in parallel
results = run_batch(adv_props, mechanisms, keys)
```

**Output:**
- Heatmaps: adversarial % vs resources
- Breakdown curves for each mechanism
- Statistical comparison

**Success Criteria:**
- ✅ 1000+ simulations complete in < 1 hour on GPU
- ✅ Results reproducible (same seed = same outcome)
- ✅ Clear visualization of mechanism differences

### 5. Learned Policies with Gradient Descent
**Goal:** Agents learn optimal strategies through experience

**Implementation:**
```python
import optax

# Initialize learnable policy
policy = LinearPolicy(input_dim=obs_dim, output_shape=action_shape)
optimizer = optax.adam(learning_rate=0.01)

# Define loss (e.g., total resources extracted)
def loss_fn(params, state, key):
    obs = get_observation(state, agent_id)
    action = policy(obs, key, params)
    new_state = apply_action(state, agent_id, action)
    return -jnp.sum(new_state.node_attrs["resources"][agent_id])

# Training loop
for epoch in range(n_epochs):
    loss, grads = jax.value_and_grad(loss_fn)(params, state, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

**Experiments:**
1. **Single agent learning** in lake environment
2. **Co-evolution** of multiple learning agents
3. **Population dynamics** (learned vs hand-coded)

**Success Criteria:**
- ✅ Learned policy outperforms random baseline
- ✅ Training converges in < 1000 episodes
- ✅ Policies generalize to unseen initial conditions

### 6. Visualization Suite
**Goal:** Make results interpretable

**Tools:**
- matplotlib for static plots
- seaborn for statistical plots
- Possibly plotly for interactive

**Plots Needed:**
1. **Population trajectories**
   - Fish over time
   - Agent resources over time
   - By mechanism and adversarial %

2. **Phase diagrams**
   - Extraction rate vs regeneration rate
   - Sustainable vs collapse regions

3. **Mechanism comparison**
   - Box plots: resources by mechanism
   - Survival curves: time to collapse
   - Heatmaps: adversarial % vs performance

4. **Network dynamics** (if time)
   - Delegation patterns in PLD
   - Representative capture in PRD

**Success Criteria:**
- ✅ Publication-quality figures
- ✅ Clear visual separation of mechanisms
- ✅ Easy to generate from raw results

## Medium Term (Next Month)

### 7. LLM Agent Integration
**Goal:** Replicate thesis red team/blue team framework

**Components:**

**Agent wrapper:**
```python
class LLMAgent:
    def __init__(self, model, system_prompt, agent_type):
        self.model = model
        self.system_prompt = system_prompt
        self.agent_type = agent_type  # "aligned" or "adversarial"

    def __call__(self, obs: jnp.ndarray, key: PRNGKey) -> jnp.ndarray:
        # Format observation as text
        obs_text = self.format_observation(obs)

        # Get LLM response
        response = self.model.generate(
            system_prompt=self.system_prompt,
            user_prompt=obs_text
        )

        # Parse action from response
        action = self.parse_action(response)
        return action
```

**Challenges:**
- **Cost:** API calls for 1000+ simulations expensive
- **Speed:** Much slower than pure JAX
- **Consistency:** LLM agents show behavioral drift

**Solutions:**
1. Cache LLM responses for repeated scenarios
2. Use smaller batch sizes (10-50 instead of 1000)
3. Implement iterative validation protocol from thesis

**Success Criteria:**
- ✅ LLM agents execute in JAX framework
- ✅ Adversarial agents show goal-directed behavior
- ✅ Results align with thesis findings

### 8. Thesis Replication Study
**Goal:** Reproduce key thesis results with this framework

**Experiments:**
1. **Baseline validation**
   - Perfect information environment
   - Verify agent goal alignment
   - Measure red team sub-optimality

2. **Main experiments**
   - Noisy information environment
   - Sweep adversarial proportions
   - Compare PDD, PRD, PLD

3. **Framing experiments**
   - Test different adversarial prompts
   - Measure consistency of destructive behavior
   - Identify effective cognitive frames

**Success Criteria:**
- ✅ Results qualitatively similar to thesis
- ✅ PLD advantage at moderate adversarial %
- ✅ Breakdown thresholds in expected range (70-90%)

### 9. Performance Optimization
**Goal:** Push JAX parallelization to limits

**Optimizations:**
1. **Larger batches:** Test 256, 512, 1024 parallel sims
2. **pmap for multi-GPU:** If available
3. **XLA flags:** Experiment with optimization flags
4. **Memory efficiency:** Reduce intermediate allocations

**Profiling:**
```bash
# JAX profiler
jax.profiler.start_trace("./logs")
run_simulation(...)
jax.profiler.stop_trace()

# Analyze with TensorBoard
tensorboard --logdir=./logs
```

**Success Criteria:**
- ✅ Identify bottlenecks
- ✅ 100x speedup on GPU (vs sequential CPU)
- ✅ Saturate GPU utilization (>80%)

## Long Term (Next 3 Months)

### 10. Mechanism Design Explorations
**Goal:** Use framework to test new collective intelligence mechanisms

**Ideas:**
1. **Quadratic voting** with resource constraints
2. **Futarchy** (prediction markets for decisions)
3. **Sortition** (random selection of decision-makers)
4. **Conviction voting** (time-weighted preferences)

**Experiments:**
- Compare to PDD, PRD, PLD baselines
- Test resilience to adversaries
- Measure efficiency (resources, decisions)

### 11. Multi-Domain Applications
**Goal:** Demonstrate generality beyond fishing/democracy

**Domains:**
1. **Market dynamics**
   - Traders, market makers, regulators
   - Price formation under manipulation

2. **Epidemic modeling**
   - Susceptible, Infected, Recovered
   - Testing vaccination strategies

3. **Traffic flow**
   - Drivers, routes, congestion
   - Testing routing algorithms

4. **Social networks**
   - Users, content, influence
   - Testing moderation policies

### 12. Publication & Sharing
**Goal:** Make research reproducible and extensible

**Outputs:**
1. **Paper:** "A JAX-Native Framework for Testing Collective Intelligence"
2. **Documentation:** Full API reference
3. **Examples:** Jupyter notebooks for common use cases
4. **Package:** PyPI release for easy installation

## Open Questions

### Technical
1. **How to handle dynamic graphs?** (agents join/leave)
2. **How to model communication delays?** (async actions)
3. **How to integrate continuous and discrete actions?**

### Research
1. **What makes PLD resilient?** (delegation? performance tracking? both?)
2. **Can we prove convergence properties?** (for learned policies)
3. **What's the optimal adversarial strategy?** (game-theoretic analysis)

### Practical
1. **How to deploy for real-world use?** (web interface? API?)
2. **How to validate against human data?** (lab experiments?)
3. **How to scale to 1000+ agents?** (current: 10-100)

## Decision Points

### If GPU Shows <10x Speedup
**Options:**
1. Profile to identify bottleneck
2. Try larger batch sizes (256+)
3. Test on cloud GPU (A100, H100)
4. Consider CPU-only development path

### If Democratic Mechanisms Don't Separate
**Options:**
1. Increase environmental pressure (lower regeneration)
2. Make adversaries more sophisticated
3. Add more complex decision spaces (10+ portfolios)
4. Focus on single-mechanism deep dive (PLD only)

### If LLM Integration is Too Slow/Expensive
**Options:**
1. Use LLMs for initial policy discovery only
2. Distill LLM behavior into fast policies
3. Focus on learned policies (gradient descent)
4. Use smaller, local models (Llama, Mistral)

---

**Current Priority:** GPU benchmarking → Democratic mechanism tuning → Parallel sweeps

**Timeline:**
- Week 1: GPU + mechanism tuning
- Week 2-3: Parallel sweeps + visualization
- Week 4: Learned policies + thesis replication
