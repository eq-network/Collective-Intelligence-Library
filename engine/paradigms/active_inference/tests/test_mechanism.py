"""
Behavioral acceptance tests — the V5 mechanism signatures (capture-the-mechanism,
not bit-exact). Each asserts a qualitative phenomenon the paper claims, run through
the real scan tier on real Gaussian belief nets.

  1. Contraction      — plain fusion collapses structural diversity to consensus.
  2. Rivals hold      — represented rivals keep divergence under full contact.
  3. Kuhn cycle       — world flip -> q(rival) climbs, entropy of q(m) peaks, resolves.
  4. Crisis signature — reliability gate produces an INTERIOR posterior-variance max.
  5. Controls         — a never-shifting world produces no crisis.
  6. Structure learner— closed-form BMR reduction score runs per agent.
  7. Sweep            — the whole paradigm vmaps over seeds as one program.

Run: python -m pytest engine/paradigms/active_inference/tests/test_mechanism.py -q
"""
import dataclasses

import jax
import jax.numpy as jnp
from jax import random

from core.scan import run_scan, run_scan_batch
from engine.paradigms.active_inference.agents import (
    ActiveInferenceAgent, StructureLearnerAgent,
)
from engine.paradigms.active_inference.schema import q_of_m, PI, LOGW
from engine.paradigms.active_inference.primitives import info_cov, entropy
from engine.paradigms.active_inference.environments import (
    two_camp_contraction, represented_rivals, kuhn_cycle, conflict_single_agent,
)


def test_contraction_collapses_structural_diversity():
    """K=1 plain fusion: two camps with different wirings converge — the off-diagonal
    coupling that distinguishes them averages away (V5 Result 1)."""
    n = 6
    cfg, state = two_camp_contraction(n_per_camp=n, coupling=0.8)
    round_fn = ActiveInferenceAgent(cfg).round_fn()

    def camp_gap(s):
        coup = s.node_attrs[PI][:, 0, 0, 1]      # off-diagonal coupling per agent
        return jnp.abs(coup[:n].mean() - coup[n:].mean())

    assert float(camp_gap(state)) > 0.5          # camps start far apart in wiring
    final, _ = run_scan(round_fn, state, 40, random.PRNGKey(0))
    assert float(camp_gap(final)) < 1e-2         # contraction: diversity gone


def test_represented_rivals_hold_divergence():
    """K=2, full contact, theory-laden attention: each camp keeps its own paradigm —
    q(m) settles at a persistent gap, not consensus (V5 Result 2)."""
    n = 6
    cfg, state = represented_rivals(n_per_camp=n)
    round_fn = StructureLearnerAgent(cfg).round_fn()

    def gap(s):
        q1 = q_of_m(s)[:, 1]                      # P(candidate 1) per agent
        return q1[n:].mean() - q1[:n].mean()      # camp B - camp A

    _, trace = run_scan(round_fn, state, 200, random.PRNGKey(0), trace_fn=gap)
    assert float(trace[-1]) > 0.5                 # divergence held at t=200
    assert float(jnp.mean(trace[-50:])) > 0.5     # stable, not a slow leak


def test_kuhn_cycle():
    """K=2 population, world flips at t_shift: normal science -> crisis (entropy peak)
    -> revolution -> new normal (V5 Result 2 / Fig 4)."""
    cfg, state, T = kuhn_cycle(n=12, T=120)
    round_fn = StructureLearnerAgent(cfg).round_fn()
    tshift = T // 2

    def readout(s):
        q1 = q_of_m(s)[:, 1].mean()               # pop. P(rival)
        ent = entropy(s.node_attrs[LOGW]).mean()  # pop. entropy of q(m)
        return jnp.array([q1, ent])

    _, trace = run_scan(round_fn, state, T, random.PRNGKey(0), trace_fn=readout)
    q1, ent = trace[:, 0], trace[:, 1]

    assert float(q1[tshift - 1]) < 0.2            # normal science (committed to incumbent)
    assert float(q1[-1]) > 0.8                    # new normal (committed to rival)
    # crisis = a torn state AFTER the flip, entropy peak above the committed level
    assert float(jnp.max(ent[tshift:])) > float(ent[tshift - 1]) + 0.2
    assert float(ent[-1]) < 0.3                   # resolved


def test_reliability_gate_interior_variance_max():
    """A channel turns contradictory: WITH the Student-t gate the posterior variance
    has an interior maximum (doubt rises then resolves); a plain Gaussian's variance
    is monotone non-increasing (V5 Fig 8)."""
    def run(use_gate):
        cfg, state, T, _tc = conflict_single_agent(use_gate=use_gate)
        round_fn = ActiveInferenceAgent(cfg).round_fn()

        def var0(s):
            return info_cov(s.node_attrs[PI][0, 0])[0, 0]   # variance of conflicted coord

        _, trace = run_scan(round_fn, state, T, random.PRNGKey(0), trace_fn=var0)
        return trace

    v_gate, v_plain = run(True), run(False)

    peak = int(jnp.argmax(v_gate))
    assert 0 < peak < len(v_gate) - 1                       # interior maximum
    assert float(v_gate[peak]) > float(v_gate[0])
    assert float(v_gate[peak]) > float(v_gate[-1])          # rises then resolves
    assert bool(jnp.all(jnp.diff(v_plain) <= 1e-4))         # plain Gaussian: monotone down


def test_control_never_shift_no_crisis():
    """A world that never flips produces no crisis: q(rival) never rises."""
    cfg, state, T = kuhn_cycle(n=12, T=120)
    cfg2 = dataclasses.replace(cfg, phi_after=cfg.phi_before, t_shift=10 ** 9)
    round_fn = StructureLearnerAgent(cfg2).round_fn()
    _, trace = run_scan(round_fn, state, T, random.PRNGKey(0),
                        trace_fn=lambda s: q_of_m(s)[:, 1].mean())
    assert float(jnp.max(trace[10:])) < 0.2


def test_structure_learner_bmr_score_runs():
    """The structure learner's closed-form BMR reduction check runs per agent."""
    cfg, state, T = kuhn_cycle(n=8, T=20)
    agent = StructureLearnerAgent(cfg)
    final, _ = run_scan(agent.round_fn(), state, T, random.PRNGKey(0))
    dF = agent.bmr_edge_score(final, candidate=0, edge=(0, 1))
    assert dF.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(dF)))


def test_paradigm_vmaps_over_seeds_one_program():
    """The whole paradigm runs as one vmap(scan(...)) program over seeds — the
    'thousands of repeated simulations' win, now for active inference."""
    cfg, state, T = kuhn_cycle(n=8, T=60)
    round_fn = StructureLearnerAgent(cfg).round_fn()
    B = 32
    keys = random.split(random.PRNGKey(0), B)
    _, traces = run_scan_batch(round_fn, lambda k: state, T, keys,
                               trace_fn=lambda s: q_of_m(s)[:, 1].mean())
    assert traces.shape == (B, T)
    assert bool(jnp.all(traces[:, -1] > 0.7))               # every seed completes the flip


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("all mechanism tests passed")
