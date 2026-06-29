"""
Tests for the Commons Harvest substrate: regrowth-tier / window-sum correctness, movement / wall /
zap mechanics, harvest accounting, the registry + EnvSpec surface, jit/vmap safety, and the
headline acceptance test — the canonical tragedy reproduction (greedy depletes; restraint sustains).
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from cilib.environments import make_env, list_envs
from cilib.environments.commons_harvest import dynamics as dyn
from cilib.environments.commons_harvest.config import HarvestConfig
from cilib.environments.commons_harvest.state import make_state


def _state(cfg, pos, apples=None, action=None, frozen=None):
    """Build a state then override the fields a unit test wants to control."""
    st = make_state(cfg, jr.PRNGKey(0))
    st = st.update_node_attrs("pos", jnp.asarray(pos, jnp.int32))
    if apples is not None:
        st = st.update_global_attr("apples", jnp.asarray(apples, jnp.float32))
    if action is not None:
        st = st.update_node_attrs("action", jnp.asarray(action, jnp.int32))
    if frozen is not None:
        st = st.update_node_attrs("frozen", jnp.asarray(frozen, jnp.int32))
    return st


# ----------------------------------------------------------------------------- registry / contract

def test_registry_and_config_overrides():
    assert "commons_harvest" in list_envs()
    env = make_env("commons_harvest", n_agents=8, grid=(12, 12), policy="sustainable")
    assert env.config.n_agents == 8
    assert env.config.height == 12 and env.config.width == 12
    assert env.config.policy == "sustainable"


def test_run_batch_shapes_and_evaluate():
    env = make_env("commons_harvest", n_agents=6, grid=(12, 12))
    finals, trs = env.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=20)
    assert trs["apples_total"].shape == (3, 20)
    assert trs["harvest"].shape == (3, 20, 6)
    # evaluate on a single run's trace (slice off the seed axis)
    one = {k: v[0] for k, v in trs.items()}
    scores = env.evaluate(one)
    assert set(scores) == {"survival_time", "total_gain", "equality", "final_apples"}


# ----------------------------------------------------------------------------- window sum / regrow

def test_window_sum_radius1():
    g = jnp.zeros((5, 5)).at[2, 2].set(1.0)
    ws = dyn._window_sum(g, 1)
    assert float(ws[2, 2]) == 1.0          # the 3x3 around the centre each see the single apple
    assert float(ws[1, 1]) == 1.0
    assert float(ws[0, 0]) == 0.0          # outside the window
    assert float(jnp.sum(ws)) == 9.0


def test_regrow_never_from_empty():
    """Zero-neighbour cells have p=0: a fully empty orchard never spontaneously regrows."""
    cfg = HarvestConfig(height=9, width=9, n_agents=1)
    regrow = dyn.make_regrow(cfg)
    st = _state(cfg, [[4, 4]], apples=np.zeros((9, 9)))
    for s in range(40):
        st = st.update_global_attr("rng_key", jr.PRNGKey(s))
        st = regrow(st)
    assert float(jnp.sum(st.global_attrs["apples"])) == 0.0


def test_regrow_only_near_existing_apples():
    """A dense patch can regrow into adjacent empty cells; far-away empties (0 neighbours) cannot."""
    cfg = HarvestConfig(height=11, width=11, n_agents=1, p_regen_high=1.0, p_regen_mid=1.0,
                        p_regen_low=1.0)
    regrow = dyn.make_regrow(cfg)
    apples = np.zeros((11, 11))
    apples[4:7, 4:7] = 1.0                 # a 3x3 block in the interior
    st = _state(cfg, [[1, 1]], apples=apples)
    st = st.update_global_attr("rng_key", jr.PRNGKey(0))
    st = regrow(st)
    out = np.asarray(st.global_attrs["apples"])
    assert out[3, 3] > 0.0                 # adjacent to the block (has neighbours) -> grows at p=1
    assert out[9, 9] == 0.0                # isolated corner, no neighbours -> never grows


# ----------------------------------------------------------------------------- movement / walls

def test_move_north_and_wall_block():
    cfg = HarvestConfig(height=10, width=10, n_agents=2)
    move = dyn.make_move(cfg)
    # agent0 in open interior moving N; agent1 at the top interior row moving N into the wall
    st = _state(cfg, [[5, 5], [1, 4]], action=[1, 1])
    out = move(st)
    pos = np.asarray(out.node_attrs["pos"])
    assert tuple(pos[0]) == (4, 5)         # moved north
    assert tuple(pos[1]) == (1, 4)         # blocked by the periphery wall -> stayed


def test_frozen_agent_does_not_move():
    cfg = HarvestConfig(height=10, width=10, n_agents=1)
    move = dyn.make_move(cfg)
    st = _state(cfg, [[5, 5]], action=[3], frozen=[2])   # wants to move E but is frozen
    out = move(st)
    assert tuple(np.asarray(out.node_attrs["pos"])[0]) == (5, 5)


# ----------------------------------------------------------------------------- harvest

def test_harvest_removes_apple_and_splits_on_shared_cell():
    cfg = HarvestConfig(height=10, width=10, n_agents=2)
    harvest = dyn.make_harvest(cfg)
    apples = np.zeros((10, 10)); apples[5, 5] = 1.0
    st = _state(cfg, [[5, 5], [5, 5]], apples=apples)
    out = harvest(st)
    last = np.asarray(out.node_attrs["last_harvest"])
    assert pytest.approx(last[0], abs=1e-6) == 0.5        # one apple split between two agents
    assert pytest.approx(last[1], abs=1e-6) == 0.5
    assert float(out.global_attrs["apples"][5, 5]) == 0.0  # cell emptied


# ----------------------------------------------------------------------------- zap / sanction

def test_zap_freezes_nearby_agent():
    cfg = HarvestConfig(height=12, width=12, n_agents=2, zap_range=4, freeze_steps=25)
    sanction = dyn.make_sanction(cfg)
    st = _state(cfg, [[5, 5], [5, 7]], action=[5, 0], frozen=[0, 0])   # agent0 zaps; agent1 in range
    out = sanction(st)
    frozen = np.asarray(out.node_attrs["frozen"])
    assert frozen[1] == 25                 # hit -> frozen
    assert frozen[0] == 0                  # zapper itself unaffected


def test_zap_out_of_range_no_freeze():
    cfg = HarvestConfig(height=20, width=20, n_agents=2, zap_range=3)
    sanction = dyn.make_sanction(cfg)
    st = _state(cfg, [[2, 2], [2, 10]], action=[5, 0], frozen=[0, 0])  # distance 8 > range
    out = sanction(st)
    assert np.asarray(out.node_attrs["frozen"])[1] == 0


# ----------------------------------------------------------------------------- jit / pytree safety

def test_round_is_jit_safe():
    cfg = HarvestConfig(n_agents=6, height=10, width=10)
    round_fn = dyn.make_round(cfg)
    st = make_state(cfg, jr.PRNGKey(0))
    out = jax.jit(lambda s, k: round_fn(s, 0, k))(st, jr.PRNGKey(1))
    assert out.global_attrs["apples"].shape == (10, 10)
    assert out.node_attrs["pos"].shape == (6, 2)


# ----------------------------------------------------------------------------- ACCEPTANCE: tragedy

def _summary(policy, seeds=4, T=300):
    env = make_env("commons_harvest", n_agents=12, grid=(16, 16), policy=policy)
    _, trs = env.run_batch(jr.PRNGKey(0), n_seeds=seeds, n_steps=T)
    apples = np.asarray(trs["apples_total"])              # (S, T)
    alive = apples > 1.0
    survival = np.where(alive.all(1), T, np.argmax(~alive, axis=1)).mean()
    return survival, apples.mean()


def test_acceptance_tragedy_vs_sustained():
    """The canonical MAS result: unrestrained harvesting collapses the commons; restraint sustains
    it. (Perolat & Leibo 2017; SocialJax 2025.)"""
    surv_greedy, stock_greedy = _summary("greedy")
    surv_sustain, stock_sustain = _summary("sustainable")
    assert surv_sustain > surv_greedy          # restraint keeps the commons viable longer
    assert stock_sustain > stock_greedy         # and at a higher standing stock
