"""
Base agent interface.

Agent: Observation → Action

That's it.
"""
from typing import Dict, Any, Callable
from abc import ABC, abstractmethod

Observation = Dict[str, Any]
Action = Dict[str, Any]
AgentPolicy = Callable[[Observation, int], Action]


class Agent(ABC):
    """Base agent: observation → action."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """observation → action"""
        pass

    def __call__(self, observation: Observation, agent_id: int) -> Action:
        """Make agent callable as policy function."""
        return self.act(observation)
