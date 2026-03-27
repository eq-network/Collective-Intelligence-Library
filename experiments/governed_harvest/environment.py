"""
Governed Commons Harvest Environment.

Composes through Mycorrhiza's core abstractions:
- GraphState for all state
- Transform pipeline for stepping
- Environment ABC for the run loop

Adapted from SocialJax Commons Harvest Open (Guo et al., 2025).
"""
import jax.numpy as jnp
import jax.random as jr

from core.environment import Environment
from core.graph import GraphState

from .state import create_initial_state
from .transforms import make_step_transform


class GovernedHarvestEnv(Environment):
    """Common-pool resource game with swappable governance mechanisms.

    The environment is fully defined by:
    1. Initial GraphState (from state.py)
    2. A composed step transform (from transforms.py)

    No need to implement get_observation_for_agent or apply_actions —
    the transform pipeline handles everything.
    """

    def __init__(self, mechanism: str = "pdd", n_agents: int = 100,
                 n_adversarial: int = 25, seed: int = 42,
                 representative_mask: jnp.ndarray = None, **kwargs):
        state = create_initial_state(
            n_agents=n_agents,
            n_adversarial=n_adversarial,
            seed=seed,
            **kwargs,
        )

        # For PRD, generate representative mask if not provided
        if mechanism == "prd" and representative_mask is None:
            key = jr.PRNGKey(seed + 1000)
            n_reps = max(5, n_agents // 10)
            rep_indices = jr.choice(key, n_agents, shape=(n_reps,), replace=False)
            representative_mask = jnp.zeros(n_agents, dtype=jnp.float32)
            representative_mask = representative_mask.at[rep_indices].set(1.0)

        step_transform = make_step_transform(
            mechanism=mechanism,
            representative_mask=representative_mask,
        )

        super().__init__(initial_state=state, step_transform=step_transform)
        self.mechanism = mechanism

    def is_terminated(self) -> bool:
        step = self.state.global_attrs["step"]
        max_steps = self.state.global_attrs["max_steps"]
        resource = self.state.global_attrs["resource_level"]
        return bool(step >= max_steps) or bool(resource < 1.0)
