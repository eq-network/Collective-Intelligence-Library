"""Equivalence gate: the ``compile_pipeline``-composed round must be bit-identical to the
``sequential``-composed round.

``compile_pipeline`` derives execution order from each step's declared reads/writes and runs
provably-independent steps (here: local_health / fit / reward) as one parallel batch. That must
not change the result. We assert exact array equality over a multi-step ``run_scan`` for both a
frozen-W regime (monocentric) and a learnable-W regime (endogenous, which adds affiliation_update
and so exercises the AFFIL write -> accumulate dependency).
"""
import numpy as np
import jax.random as jr

from cilib.core.category import sequential
from cilib.core.pipeline import compile_pipeline
from cilib.core.scan import run_scan
from cilib.paradigms import polycentric as P
from cilib.paradigms.polycentric.transforms import round_factories
from cilib.paradigms.polycentric.schema import RNG


def _round(pipeline):
    def round_fn(state, t, key):
        return pipeline(state.update_global_attr(RNG, key))
    return round_fn


def _assert_states_equal(a, b):
    for container in ("node_attrs", "adj_matrices", "global_attrs"):
        da, db = getattr(a, container), getattr(b, container)
        assert set(da) == set(db), container
        for k in da:
            assert np.array_equal(np.asarray(da[k]), np.asarray(db[k])), f"{container}[{k}]"


def _check(governance):
    cfg = P.make_config(governance)
    steps = [f(cfg) for f in round_factories(cfg)]
    seq_round = _round(sequential(*steps))
    comp_round = _round(compile_pipeline(steps))

    state0 = P.make_state(cfg, jr.PRNGKey(0))
    key = jr.PRNGKey(1)
    final_seq, _ = run_scan(seq_round, state0, 60, key)
    final_comp, _ = run_scan(comp_round, state0, 60, key)
    _assert_states_equal(final_seq, final_comp)


def test_compiled_equals_sequential_endogenous():
    _check("endogenous")


def test_compiled_equals_sequential_monocentric():
    _check("monocentric")


def test_compiler_batches_independent_steps():
    """Sanity: the compiler actually finds the local_health|fit|reward parallel batch
    (otherwise equivalence would be trivially true via a fully-serial schedule)."""
    cfg = P.make_config("endogenous")
    compiled = compile_pipeline([f(cfg) for f in round_factories(cfg)])
    batch_sizes = [len(b) for b in compiled._batches]
    assert max(batch_sizes) >= 2, batch_sizes      # at least one genuine parallel batch


if __name__ == "__main__":
    test_compiled_equals_sequential_endogenous()
    test_compiled_equals_sequential_monocentric()
    test_compiler_batches_independent_steps()
    print("ok  pipeline equivalence")
