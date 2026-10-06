"""The Ledger Society on ``EnvSpec.run_grid``: traced channel dials, the swept
defense plan, the streaming metrics, and the influence-from-birth readout.

``traced_channels`` off is the pre-2026-10-06 model by construction (the same
expressions), checked once against a T=400 snapshot of every run path when it
was added (docs/run-grid-design.md). What these tests hold: the flag adds no
change a single rollout can see, a grid cell is exactly the spec built with its
value, and the helpers agree with the trace-based metrics they stand in for.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from cilib.environments import make_env
from cilib.environments.ledger_society.config import CHANNELS
from cilib.environments.ledger_society.influence import (influence_from_birth,
                                                         onset_grid)
from cilib.environments.ledger_society.metrics import SHARES, make_reducers

KEY, S, T = jr.PRNGKey(3), 3, 60
LEVEL = [0.05, 0.02, 0.5, 0.1]
COUPLED = dict(reach_per_spend=4.0, attention_to_ballots=2.0, regime_rate=0.02)


def _same(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)


def test_channel_globals_exist_only_when_traced():
    off = make_env("ledger_society").init_fn(KEY).global_attrs
    on = make_env("ledger_society", traced_channels=True).init_fn(KEY).global_attrs
    assert not {c + "_now" for c in CHANNELS} & set(off)
    assert {c + "_now" for c in CHANNELS} <= set(on)
    assert "t_on_now" not in on
    assert "t_on_now" in make_env("ledger_society", policy_horizon=5).init_fn(KEY).global_attrs


@pytest.mark.parametrize("cfg", [{}, COUPLED], ids=["default", "coupled"])
def test_traced_channels_leave_a_single_rollout_unchanged(cfg):
    """The batched paths move at float32 rounding when the dials are lifted (why
    the flag exists); a single rollout does not, here or in the T=400 check."""
    _, off = make_env("ledger_society", **cfg).run(KEY, T)
    _, on = make_env("ledger_society", traced_channels=True, **cfg).run(KEY, T)
    _same(on, off)


@pytest.mark.parametrize("dial", CHANNELS)
def test_grid_cell_is_the_spec_built_with_its_value(dial):
    values = (0.0, 0.5 * getattr(make_env("ledger_society").config, dial), 0.03)
    env = make_env("ledger_society", traced_channels=True)
    reducers = make_reducers(env.config)
    got = env.run_grid(KEY, S, T, {dial + "_now": jnp.array(values)}, reducers)
    for g, v in enumerate(values):
        want = make_env("ledger_society", traced_channels=True, **{dial: v}
                        ).run_reduced_batch(KEY, S, T, reducers)
        _same(jax.tree_util.tree_map(lambda x: x[g], got), want)


def test_reducers_match_the_trace_metrics():
    env = make_env("ledger_society", **COUPLED)
    _, trace = env.run_batch(KEY, S, T)
    _, _, red = env.run_reduced_batch(KEY, S, T, make_reducers(env.config))
    for name in (*SHARES, "composite"):
        np.testing.assert_allclose(red[name], jax.vmap(env.metrics[name])(trace), rtol=1e-5)


def test_onset_grid_ties_the_plan_to_the_onset_dial():
    g = onset_grid([0, 7, T], LEVEL, T)
    assert g["policy_plan"].shape == (3, T, 4) and g["t_on_now"].dtype == jnp.int32
    assert float(jnp.abs(g["policy_plan"][1, :7]).sum()) == 0.0
    np.testing.assert_array_equal(g["policy_plan"][1, 7], np.float32(LEVEL))
    assert float(jnp.abs(g["policy_plan"][2]).sum()) == 0.0      # never switches on
    with pytest.raises(ValueError, match="plan column"):
        onset_grid([0], [0.1, 0.2], T)


def test_composite_at_onset_is_the_state_before_the_defense():
    """at_step against the trace, and the plan really starts at its onset: cells
    agree with the never-defended cell up to t_on - 1 and not after."""
    env = make_env("ledger_society", traced_channels=True, policy_horizon=T, **COUPLED)
    onsets = [10, 30, T]
    _, trace, red = env.run_grid(KEY, S, T, onset_grid(onsets, LEVEL, T),
                                 make_reducers(env.config), trace_fn=env.trace_fn)
    comp = (trace["human_income_share"] + trace["human_attention_share"]
            + trace["human_power_share"]) / 3.0                       # (G, S, T)
    for g, on in enumerate(onsets):
        np.testing.assert_allclose(red["composite_at_onset"][g], comp[g, :, on - 1], rtol=1e-6)
    for g, on in enumerate(onsets[:2]):
        _same(trace["wealth"][g, :, :on], trace["wealth"][2, :, :on])
        assert not np.array_equal(trace["wealth"][g, :, on], trace["wealth"][2, :, on])


def test_influence_from_birth_is_one_compile_per_spec_and_responds():
    env = make_env("ledger_society", traced_channels=True, policy_horizon=T)
    grid = onset_grid([0, T], LEVEL, T)
    infl = influence_from_birth(env, KEY, S, T, grid, delta=0.05)
    assert infl.shape == (2, S)
    assert len(env._programs) == 1
    # at defaults humans hold most ballots, so a shift in what they want moves
    # the enacted rate the same way (a sign check, not a size claim)
    assert bool(jnp.all(infl > 0.0))
