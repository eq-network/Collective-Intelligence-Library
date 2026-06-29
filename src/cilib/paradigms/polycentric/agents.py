"""
Agent wrapper for the polycentric paradigm (active_inference PureAgent pattern).

State lives in the GraphState, not the agent. An agent is a factory for a ``round_fn`` compatible
with ``core.scan.run_scan`` / ``run_scan_batch``. Active-inference reading: ``quota`` is the
institution's policy prior, ``affiliation`` the Markov blanket; a frozen-precise central
controller (monocentric, freeze_affiliation) is the calcification limit (over-precise prior,
no learning), an endogenous ensemble keeps a high affiliation learning rate per niche.
"""
from __future__ import annotations

from cilib.core.scan import RoundFn
from .schema import PolyConfig
from .transforms import make_round


class PolycentricAgent:
    def __init__(self, cfg: PolyConfig):
        self.cfg = cfg

    def round_fn(self) -> RoundFn:
        return make_round(self.cfg)
