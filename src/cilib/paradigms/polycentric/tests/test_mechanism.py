"""
Behavioral acceptance tests for the polycentric commons paradigm.

Qualitative assertions (direction / ordering), in the style of active_inference's
test_mechanism.py — not bit-exact values. They pin the phenomena the scientific claims rest on:
- tragedy is the default (atomized collapses);
- enforcement is load-bearing (monitoring-off ablation collapses);
- governed regimes survive;
- endogenous agents form institutions that recover the latent sub-communities;
- monocentric trades fit for control, and the endogenous-vs-monocentric FIT gap grows with
  heterogeneity (the headline EI∧fit signature).

Run: python -m pytest engine/paradigms/polycentric/tests/test_mechanism.py -q
"""
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from cilib.paradigms import polycentric as P
from cilib.analysis import causal_emergence as ce

KEY = jr.PRNGKey(0)
T = 150


def _run(governance, het=1.5, monitoring=None, capture=0.0, seed=0, T=T):
    cfg = P.make_config(governance, heterogeneity=het, monitoring=monitoring, capture=capture)
    final, tr = P.run(cfg, jr.PRNGKey(seed), T=T)
    return cfg, final, tr


def _final_R(tr):
    return float(np.asarray(tr["resource"])[-1])


def _mean_fit(tr, last=50):
    return float(np.asarray(tr["fit"])[-last:].mean())


def test_round_runs_and_preserves_structure():
    cfg, final, tr = _run("endogenous", T=50)
    assert np.asarray(tr["resource"]).shape == (50,)
    assert np.asarray(tr["harvest"]).shape == (50, cfg.n_agents)
    assert np.all(np.isfinite(np.asarray(tr["resource"])))
    assert np.all(np.isfinite(np.asarray(tr["harvest"])))


def test_run_batch_shapes_and_reproducible():
    cfg = P.make_config("endogenous")
    finals, trs = P.run_batch(cfg, KEY, n_seeds=4, T=40)
    assert np.asarray(trs["resource"]).shape == (4, 40)
    finals2, trs2 = P.run_batch(cfg, KEY, n_seeds=4, T=40)
    assert np.allclose(np.asarray(trs["resource"]), np.asarray(trs2["resource"]))  # deterministic


def test_atomized_collapses():
    _, _, tr = _run("atomized")
    assert _final_R(tr) < 2.0                       # tragedy of the commons


def test_governed_regimes_survive():
    for gov in ("monocentric", "fixed_poly", "endogenous"):
        _, _, tr = _run(gov)
        assert _final_R(tr) > 20.0, gov             # commons sustained


def test_enforcement_is_load_bearing():
    """Policer-ablation: turn monitoring OFF on the endogenous regime -> collapse (C6/H0)."""
    _, _, tr = _run("endogenous", monitoring=False)
    assert _final_R(tr) < 2.0


def test_endogenous_recovers_sub_communities():
    cfg, final, tr = _run("endogenous")
    Wbar = np.asarray(final.global_attrs[P.AFFIL_SUM]) / T
    labels = ce.emergent_partition(Wbar)
    assert len(np.unique(labels)) == cfg.n_blocks
    blk = np.asarray(P.block_assignment(cfg))
    # emergent communities align with the latent blocks (each block internally homogeneous)
    for b in range(cfg.n_blocks):
        comm = labels[blk == b]
        assert len(np.unique(comm)) == 1, (b, comm)


def test_monocentric_trades_fit_vs_endogenous():
    """At nonzero heterogeneity the uniform central quota fits worse than emergent local quotas."""
    _, _, tr_mono = _run("monocentric", het=2.0)
    _, _, tr_endo = _run("endogenous", het=2.0)
    assert _mean_fit(tr_endo) > _mean_fit(tr_mono) + 0.2


def test_fit_gap_grows_with_heterogeneity():
    """The headline EI∧fit signature: endogenous-minus-monocentric fit gap widens with het."""
    def gap(het):
        _, _, mono = _run("monocentric", het=het)
        _, _, endo = _run("endogenous", het=het)
        return _mean_fit(endo) - _mean_fit(mono)
    gap_low = gap(0.0)
    gap_high = gap(2.0)
    assert gap_low < 0.15                            # homogeneous: central quota fits fine
    assert gap_high > gap_low + 0.3                  # heterogeneous: emergent local quotas win


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all polycentric mechanism tests passed")
