# Plans & Documentation

Comprehensive documentation for the Mycorrhiza agent system development.

## Files

### [CONTEXT.md](CONTEXT.md)
**What we built and why**
- Core agent system implementation
- Lake model (central resource hub)
- Democratic mechanisms (PDD, PRD, PLD)
- JAX parallelization benchmarks
- Key learnings and findings
- Thesis context

**Read this first** to understand what's been done.

### [ARCHITECTURE.md](ARCHITECTURE.md)
**How the system works**
- Design philosophy (functional, JAX-native)
- Core data structures (GraphState)
- Agent flow (observation → policy → action)
- JAX integration patterns (vmap, jit, scan)
- Lake model dynamics
- Democratic mechanisms implementation
- Extension points

**Read this** to understand the technical design.

### [NEXT_STEPS.md](NEXT_STEPS.md)
**What to do next**
- Immediate: GPU benchmarking, mechanism stress tests
- Short term: Parallel sweeps, learned policies, visualization
- Medium term: LLM integration, thesis replication
- Long term: New mechanisms, multi-domain applications
- Open questions and decision points

**Read this** to know what's next.

## Quick Navigation

### I want to...

**Understand what was built:**
→ Start with CONTEXT.md

**Understand how it works:**
→ Read ARCHITECTURE.md

**Know what to work on next:**
→ Check NEXT_STEPS.md

**Run the tests:**
```bash
# Simple lake model
python simple_lake_test.py

# Democratic fishing
python democratic_fishing_test.py

# JAX parallelization (GPU in WSL)
python3 test_jax_parallelization.py
```

**Set up GPU:**
```bash
# In WSL Ubuntu
pip3 install --upgrade "jax[cuda12_pip]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify
python3 -c "import jax; print(jax.devices())"
```

## Summary

### What Works ✅
- Pure functional agent system
- Central lake model (tragedy of commons)
- Democratic voting mechanisms
- JAX parallelization (1.3x on CPU, 10-100x expected on GPU)
- Concrete policies (random, learnable)

### What's Next ⏳
- GPU benchmarking in WSL
- Democratic mechanism stress testing
- Parallel adversarial sweeps
- Learned policies with gradient descent
- Thesis replication

### Key Insight
**Action semantics matter more than framework design.**
Same framework works for:
- Transfer semantics (agents exchange)
- Extraction semantics (agents consume)
- Just swap `apply_action()` implementation

## Thesis Context

This work supports:
**"Red Teaming Democracy: LLM Simulation of Institutional Resilience"**
Jonas Hallgren, Uppsala University, 2026

The JAX-native functional architecture provides the substrate for:
- Red team/blue team adversarial testing
- Parallel simulation of 1000+ scenarios
- Deterministic, reproducible results
- Fast iteration on GPU

## Status

**Last Updated:** January 22, 2026
**Current Phase:** GPU benchmarking + mechanism tuning
**Next Milestone:** Parallel adversarial sweeps

---

For questions or clarifications, check the individual files or ask Jonas.
