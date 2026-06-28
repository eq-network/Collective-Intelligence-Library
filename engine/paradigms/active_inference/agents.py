"""
Active-inference agents.

These are the OO handles the user reasons about (`Agent -> ActiveInferenceAgent
-> StructureLearnerAgent`), but they are PURE: an agent does not *hold* belief
state (that lives in GraphState.node_attrs). Instead an agent is a factory for
the pure round it implements — its ``round_fn()`` returns a
``(state, t, key) -> state`` function that ``core.scan.run_scan`` folds over time
and ``vmap``s over seeds. Contrast the effectful ``engine.agents.llm_agent``,
whose ``act`` brackets an HTTP call and must run on the eager tier.

PureAgent convention (grounded here rather than guessed earlier): a pure agent
exposes ``round_fn() -> RoundFn``. That is the whole contract — anything pure that
yields a scan-compatible round is a PureAgent.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.scan import RoundFn
from core.category import sequential

from .schema import AIConfig
from . import transforms


@runtime_checkable
class PureAgent(Protocol):
    """A side-effect-free agent: yields the pure round it implements, runnable
    inside ``core.scan.run_scan`` and ``vmap``-able over seeds."""
    def round_fn(self) -> RoundFn: ...


class ActiveInferenceAgent:
    """The baseline believer: observe -> infer -> fuse -> forget over a single
    (or fixed) Gaussian wiring. With ``cfg.track_logweights=False`` and K=1 this is
    the plain precision-weighted-averaging baseline whose fusion is a contraction.
    """

    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self._observe = transforms.make_observe(cfg)
        self._fuse = transforms.make_fuse(cfg)
        self._forget = transforms.make_forget(cfg)

    def round_fn(self) -> RoundFn:
        observe, fuse, forget = self._observe, self._fuse, self._forget
        # fuse + forget are plain Transforms (state->state); observe needs (t, key).
        post_observe = sequential(fuse, forget)

        def round(state, t, key):
            state = observe(state, t, key)
            return post_observe(state)

        return round


class StructureLearnerAgent(ActiveInferenceAgent):
    """Structure as an object of inference. Same round as the baseline, but run
    with K>=2 candidate wirings + ``track_logweights=True`` so prequential
    evidence accumulates in the log-weights and the damped hypothesis pool keeps
    rival readings of the world apart — this is what reproduces the Kuhn cycle.

    Also exposes the closed-form Bayesian-Model-Reduction check (the "reduction"
    half of structure learning): score whether the data are content to prune a
    coupling from a candidate's wiring.
    """

    def bmr_edge_score(self, state, candidate: int, edge: tuple):
        """Per-agent ΔF for pruning ``edge`` from ``candidate`` (see
        ``transforms.bmr_edge_score``). ΔF > 0 ⇒ reduction favoured."""
        return transforms.bmr_edge_score(self.cfg, state, candidate, edge)
