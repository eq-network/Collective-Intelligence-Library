"""
Causal readouts for ``EnvSpec.run_grid`` sweeps of the Ledger Society.

Two host-side helpers for the rescue-probe grid:

- ``onset_grid`` builds the swept defense plan and the matching ``t_on_now``
  bookkeeping dial from the same onsets, so the ``at_step`` reducer and the plan
  can never disagree about when the defense switches on.
- ``influence_from_birth`` is ``counterfactual.collective_influence`` per grid
  cell: shift every human's preferred tax rate by ``delta`` at birth, run the
  same grid on the same key, and take ``(y_shifted - y_base) / delta`` per cell
  and seed. It asks whether human preferences still move the policy outcome, a
  causal question the share metrics cannot answer: a share can stay high while
  the outcome no longer responds to the people holding it.

The shifted spec differs from the base one only in ``init_fn``, so it is one more
compile for the whole grid, not one per cell.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Sequence

import jax.numpy as jnp
import numpy as np

from cilib.core.graph import GraphState
from cilib.metrics.reducers import late_mean

from ..spec import EnvSpec
from .config import LedgerSocietyConfig


def onset_grid(t_on: Sequence[int], level: Sequence[float], n_steps: int) -> Dict[str, Any]:
    """``{"policy_plan": (G, n_steps, 4), "t_on_now": (G,)}`` for ``run_grid``.

    Cell ``g`` holds the defense ``level`` (the four plan columns: levy, repair,
    reach cut, sortition) from step ``t_on[g]`` on and zero before. Build the env
    with ``policy_horizon=n_steps``: the plan is read by step and a shorter plan
    repeats its last row past its end. ``t_on[g] >= n_steps`` is a cell where the
    defense never switches on.
    """
    t_on = np.asarray(t_on, dtype=np.int32)
    level = np.asarray(level, dtype=np.float32)
    if level.shape != (4,):
        raise ValueError(f"level is one value per plan column (4,), got {level.shape}")
    on = np.arange(n_steps)[None, :] >= t_on[:, None]                  # (G, T)
    plan = on[:, :, None].astype(np.float32) * level[None, None, :]
    return {"policy_plan": jnp.asarray(plan), "t_on_now": jnp.asarray(t_on)}


def shift_preferences(state: GraphState, delta: float) -> GraphState:
    """Every human's preferred tax rate (``ideal``) and declared ``position``
    raised by ``delta``. AI rows keep theirs; AI positions follow their
    delegators' ideals through ``declare_position`` from the first tick."""
    h = (state.node_types == 0).astype(jnp.float32) * delta
    state = state.update_node_attrs("ideal", state.node_attrs["ideal"] + h)
    return state.update_node_attrs("position", state.node_attrs["position"] + h)


def policy_outcome(state: GraphState):
    """The enacted rule this tick: the power-weighted median vote's tax target."""
    return state.global_attrs["policy_target"]


def influence_from_birth(env: EnvSpec, key, n_seeds: int, n_steps: int,
                         grid: Dict[str, Any], *, delta: float = 0.05,
                         frac: float = 0.25):
    """``(G, n_seeds)`` responsiveness of the late-window policy outcome to a
    common shift ``delta`` in human preferences, from paired ``run_grid`` calls
    on one key. 1 means the outcome moves one-for-one with what humans want;
    0 means human preferences no longer reach it.

    ``delta`` is a finite difference: keep it small against the 0-1 tax range, and
    expect the median vote to make the response step-like rather than smooth.
    """
    outcome = {"y": late_mean(policy_outcome, frac)}
    shifted = dataclasses.replace(
        env, init_fn=lambda k: shift_preferences(env.init_fn(k), delta))
    _, _, y0 = env.run_grid(key, n_seeds, n_steps, grid, outcome)
    _, _, y1 = shifted.run_grid(key, n_seeds, n_steps, grid, outcome)
    return (y1["y"] - y0["y"]) / delta
