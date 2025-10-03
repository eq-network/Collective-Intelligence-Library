# core/agents.py
"""
Defines the abstract base class for all agents in the simulation framework.

This module establishes the fundamental interface that every agent must implement,
ensuring that the simulation engine can interact with any agent type in a
standardized way.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, TypeAlias

# For clarity, define a type alias for an Action.
# An Action is a dictionary specifying the agent's intended changes.
Action: TypeAlias = Dict[str, Any]

class Agent(ABC):
    """
    Abstract Base Class for a simulation agent.

    Each agent has a unique ID and encapsulates its own internal state and
    decision-making logic. The core responsibility of an agent is to
    produce an `Action` when prompted by the environment.
    """
    def __init__(self, agent_id: int):
        if not isinstance(agent_id, int) or agent_id < 0:
            raise ValueError("agent_id must be a non-negative integer.")
        self.agent_id = agent_id

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Action:
        """
        The primary decision-making method for the agent.

        Based on the provided observation from the environment, the agent
        must decide on an action to take.

        Args:
            observation: A dictionary containing all the information the agent
                         can perceive from the environment in the current state.
                         This could include market signals, other agents' public
                         states, collective resource levels, etc.

        Returns:
            An Action dictionary specifying the agent's desired action.
            For example: {'vote_for_portfolio': 3} or {'delegate_to': 5}.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id})"