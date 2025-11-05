"""
Random agent.

Observation → Random Action
"""
import jax.random as jr
from typing import Callable
from engine.agents.base import Agent, Observation, Action


class RandomAgent(Agent):
    """Random policy: observation → random action."""

    def __init__(self, action_sampler: Callable[[Observation, jr.PRNGKey], Action], seed: int = 42):
        self.action_sampler = action_sampler
        self.key = jr.PRNGKey(seed)

    def act(self, observation: Observation) -> Action:
        self.key, subkey = jr.split(self.key)
        return self.action_sampler(observation, subkey)
