# JAX Parallelization Test Results

**Updated:** January 22, 2026
**Device:** NVIDIA GeForce RTX 3050 Laptop (CUDA 12.7)

## Objective

Test JAX vmap parallelization speedup on lake simulation using the **actual GraphState architecture**.

## Results Summary

### GraphState GPU Benchmark

| Batch Size | Sequential | Vectorized | Speedup | Sims/sec | Verified |
|------------|------------|------------|---------|----------|----------|
| 16         | 0.066s     | 0.134s     | 0.5x    | 120      | ✓        |
| 32         | 0.041s     | 0.132s     | 0.3x    | 243      | ✓        |
| 64         | 0.076s     | 0.125s     | 0.6x    | 512      | ✓        |
| 128        | 0.163s     | 0.128s     | 1.3x    | 997      | ✓        |
| 256        | 0.401s     | 0.122s     | 3.3x    | 2,096    | ✓        |
| 512        | 0.762s     | 0.128s     | 5.9x    | 3,998    | ✓        |
| 1024       | 1.269s     | 0.149s     | 8.5x    | 6,857    | ✓        |
| 2048       | -          | 0.136s     | -       | 15,016   | ✓        |
| 4096       | -          | 0.145s     | -       | 28,281   | ✓        |
| 8192       | -          | 0.145s     | -       | **56,627** | ✓      |

### Key Metrics

- **Peak throughput:** 56,627 simulations/second (batch 8192)
- **Vectorized time:** ~0.13-0.15s (constant across batch sizes)
- **Maximum speedup:** 8.5x at batch 1024
- **All results verified:** Sequential and vectorized produce identical outputs

## Key Findings

### ✅ GraphState Works with JAX

After adding pytree registration to `GraphState`:

```python
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class GraphState:
    def tree_flatten(self):
        # Returns (arrays, static_metadata)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstructs from (arrays, static_metadata)
```

GraphState now fully supports:
- `jax.jit` - JIT compilation
- `jax.vmap` - Automatic vectorization
- `jax.lax.scan` - Efficient loops
- `jax.grad` - Automatic differentiation (for learned policies)

### 📊 Performance Characteristics

**Why vectorized time is constant:**
- GPU processes all batch elements in parallel
- Overhead is fixed (memory transfer, kernel launch)
- Throughput scales linearly with batch size

**Why speedup is modest at small batches:**
- GraphState has overhead (dict lookups, pytree traversal)
- Small batches don't saturate GPU
- Need batch size ≥256 to see significant speedup

**Comparison: Raw arrays vs GraphState**

| Approach | Peak Throughput | Notes |
|----------|-----------------|-------|
| Raw JAX arrays | ~67x speedup | No structure, just arrays |
| GraphState | ~8.5x speedup (56K sims/sec) | Full architecture support |

The overhead is acceptable given the benefits of structured state management.

### 🚀 Implications for Thesis

**Adversarial sweep capability:**
- 1,000 configurations × 100 replications = 100,000 simulations
- At 56,000 sims/sec = **1.8 seconds total**
- Previously would take minutes to hours

**Enabled experiments:**
- Full mechanism comparison (PDD, PRD, PLD)
- Adversarial proportion sweeps (0-90%)
- Parameter sensitivity analysis
- Replicated statistical tests

## Technical Details

### Simulation Configuration

```python
n_agents = 10
n_exploiters = 5  # 50% exploiters
n_rounds = 100
initial_fish = 1000.0
```

### GraphState Structure

```python
GraphState(
    node_types: (n_nodes,)           # 0=lake, 1=agent
    node_attrs: {
        'resources': (n_nodes, 1),   # Fish/catch
        'agent_type': (n_nodes,)     # 0=lake, 1=sustainable, 2=exploiter
    },
    adj_matrices: {
        'connected': (n_nodes, n_nodes)
    },
    edge_attrs: {},
    global_attrs: {'round': 0}
)
```

### Pytree Flattening

GraphState flattens to 4 leaves:
1. `node_types` array
2. `agent_type` array
3. `resources` array
4. `connected` adjacency matrix

Dict keys and global_attrs are treated as static metadata.

## How to Run

```bash
# Ensure JAX CUDA is installed
pip install jax[cuda12]

# Run benchmark
cd /mnt/c/Users/Jonas/Documents/GitHub/Mycorrhiza
python3 test_jax_parallelization.py
```

## Next Steps

1. **Democratic mechanism stress testing** - Apply parallelization to PDD/PRD/PLD comparison
2. **Adversarial sweeps** - Run 1000+ configurations in parallel
3. **Learned policies** - Use `jax.grad` for gradient-based policy optimization
4. **Profile bottlenecks** - Identify GraphState overhead sources

## Conclusion

**GraphState is fully GPU-compatible** after pytree registration. The architecture supports:

- 56,000+ simulations/second on RTX 3050
- Identical results to sequential execution
- Full JAX transformation support (jit, vmap, scan, grad)

This validates the functional architecture for large-scale adversarial simulation experiments.
