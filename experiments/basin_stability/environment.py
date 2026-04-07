"""
Basin Stability Environment.

Composes through Mycorrhiza's core abstractions:
- GraphState for all state
- Transform pipeline for stepping
- Environment ABC for the run loop

Paper: "Basin Stability of Democratic Mechanisms Under Adversarial Pressure"
"""
from core.environment import Environment
from core.graph import GraphState

from .state import create_initial_state
from .transforms import make_step_transform


class BasinStabilityEnv(Environment):
    """Proposal-selection resource game with swappable governance.

    The environment is fully defined by:
    1. Initial GraphState (from state.py)
    2. A composed step transform (from transforms.py)

    Resource collapses if R < collapse_threshold.
    """

    def __init__(self, mechanism: str = "pdd", n_agents: int = 20,
                 n_adversarial: int = 0, seed: int = 42, **kwargs):
        state = create_initial_state(
            n_agents=n_agents,
            n_adversarial=n_adversarial,
            mechanism=mechanism,
            seed=seed,
            **kwargs,
        )

        step_transform = make_step_transform(mechanism=mechanism)

        super().__init__(initial_state=state, step_transform=step_transform)
        self.mechanism = mechanism

    def is_terminated(self) -> bool:
        step = self.state.global_attrs["step"]
        T = self.state.global_attrs["T"]
        resource = self.state.global_attrs["resource_level"]
        threshold = self.state.global_attrs["collapse_threshold"]
        return bool(step >= T) or bool(resource < threshold)
