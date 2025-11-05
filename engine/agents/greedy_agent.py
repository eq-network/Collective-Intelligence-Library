"""
Greedy agent.

(Observation, Action) → Score
Select action with max score.
"""
import jax.numpy as jnp
from typing import Callable, List
from engine.agents.base import Agent, Observation, Action


class GreedyAgent(Agent):
    """Greedy policy: select action with highest score."""

    def __init__(
        self,
        score_fn: Callable[[Observation, Action], float],
        action_generator: Callable[[Observation], List[Action]]
    ):
        self.score_fn = score_fn
        self.action_generator = action_generator

    def act(self, observation: Observation) -> Action:
        candidates = self.action_generator(observation)
        if not candidates:
            return {}

        scores = jnp.array([self.score_fn(observation, a) for a in candidates])
        return candidates[int(jnp.argmax(scores))]
