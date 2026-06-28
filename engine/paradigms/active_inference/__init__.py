"""
Active inference (information-form Gaussian) paradigm.

A population of agents on a trust graph; each agent holds Gaussian beliefs over a
set of commitments and (optionally) a posterior over rival WIRINGS of those
commitments. Beliefs fuse across the trust graph (precision-weighted averaging —
a contraction), the world flips at t_shift, and prequential evidence reorders the
candidate wirings — reproducing the Kuhn cycle.

Ported from the "paradigm shift" / changing-networked-mind model
(C:/GitHub/Paradigm_Shift_Act_Inf). The Gaussian/BMR math is vendored verbatim in
``primitives/``; the dynamics are re-expressed as pure, vectorized transforms that
run on the scan tier (``core.scan.run_scan``) and ``vmap`` over seeds.

Quickstart::

    from core.scan import run_scan
    from engine.paradigms.active_inference.environments import kuhn_cycle
    from engine.paradigms.active_inference.agents import StructureLearnerAgent
    from engine.paradigms.active_inference.schema import q_of_m
    import jax

    cfg, state, T = kuhn_cycle()
    round_fn = StructureLearnerAgent(cfg).round_fn()
    final, trace = run_scan(round_fn, state, T, jax.random.PRNGKey(0),
                            trace_fn=lambda s: q_of_m(s).mean(0))   # pop. q(m) over time
"""
from .schema import AIConfig, make_state, q_of_m
from .agents import PureAgent, ActiveInferenceAgent, StructureLearnerAgent
from . import transforms, environments, primitives
