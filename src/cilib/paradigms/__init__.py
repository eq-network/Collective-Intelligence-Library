"""
Paradigm imports — self-contained simulation paradigms that plug into the core
framework (GraphState + Transform + core.scan.run_scan).

Each paradigm is a folder providing, by convention (see README.md):
  schema.py        — the node_attrs / adj_matrices layout (the agent state)
  primitives/      — vendored pure math kernels (scenario-free)
  transforms.py    — pure, vectorized, gateable round steps
  agents.py        — PureAgent factories that assemble the round
  environments/    — create_initial_state scenario factories
  tests/           — behavioral acceptance signatures

The first paradigm is ``active_inference`` (information-form Gaussian belief
fusion + Bayesian structure learning), ported from the "paradigm shift" /
changing-networked-mind model.
"""
