"""
Config + static layout for the spatial Commons Harvest substrate.

Faithful to the Melting Pot / SocialJax ``Commons Harvest`` substrate (Perolat & Leibo 2017;
Guo et al. 2025): a grid of regrowable apples where an empty cell respawns an apple with a
probability that *rises with the local apple density* — so harvesting the last apples in a region
drives its regrowth probability to zero (permanent local depletion). That density-dependence is
the tragedy mechanism.

The config is a frozen dataclass closed over by the transforms (never stored in ``GraphState`` —
``core.scan`` requires ``global_attrs`` to be either dynamic arrays or static aux). ``layout``
returns the *static* grid masks (computed with numpy at trace time) so nothing dynamic-shaped
(``flatnonzero``) ever runs inside the vmapped ``init_fn`` / scanned round.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class HarvestConfig:
    n_agents: int = 16
    height: int = 18
    width: int = 18

    # perception (egocentric Chebyshev window)
    view_radius: int = 5

    # regrowth — # apples within Chebyshev ``regrow_radius`` (excluding the cell) sets respawn prob
    regrow_radius: int = 2
    p_regen_high: float = 0.025   # >= 3 apples nearby
    p_regen_mid: float = 0.005    # == 2
    p_regen_low: float = 0.001    # == 1
    #                                == 0  -> 0 (permanent depletion)

    # initial seeding
    init_apple_frac: float = 0.20  # fraction of orchard cells that start with an apple

    # enforcement primitive (the zap / fine beam → freezes a hit agent for ``freeze_steps``)
    zap_enabled: bool = True
    zap_range: int = 4
    freeze_steps: int = 25

    # agent policy for the governance-agnostic substrate (the acceptance-test reference behaviours).
    # "greedy" harvests every apple it reaches (drives the tragedy); "sustainable" only harvests
    # from cells whose local density is high enough to keep regrowing (leaves seed apples).
    policy: str = "greedy"          # {"greedy", "sustainable"}
    sustainable_density: int = 3    # min apple neighbours (within regrow_radius, excl. centre) for a
    #                                 sustainable agent to harvest a cell — keeps it in the top
    #                                 regrowth tier (>=3 -> p_regen_high) so the patch keeps growing
    learning_rate: float = 0.1

    # action space: 0 noop, 1 N(r-1), 2 S(r+1), 3 E(c+1), 4 W(c-1), 5 zap
    n_actions: int = 6

    @property
    def grid(self) -> Tuple[int, int]:
        return (self.height, self.width)


def layout(cfg: HarvestConfig):
    """Static layout arrays (numpy): ``(wall, orchard, interior_flat)``.

    ``wall`` is the periphery (the *Open* variant); ``orchard`` is the interior where apples grow
    and agents live; ``interior_flat`` are the flat indices of orchard cells, used to place agents
    on distinct cells with ``jax.random.choice(..., replace=False)`` (a fixed-shape, vmap-safe op).
    """
    H, W = cfg.height, cfg.width
    wall = np.zeros((H, W), dtype=bool)
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True
    orchard = ~wall
    interior_flat = np.flatnonzero(orchard.ravel())
    return wall, orchard, interior_flat
