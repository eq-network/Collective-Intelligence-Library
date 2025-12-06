"""
Base agent interface.

Agent: Observation → Action

That's it.

NOTE: This module re-exports from core.agents to maintain a single source of truth.
The canonical Agent protocol is defined in core/agents.py.
"""
from typing import Callable

# Re-export from canonical source
from core.agents import Agent, Observation, Action

# Additional type alias for policy functions
AgentPolicy = Callable[[Observation, int], Action]


class StatefulAgent(Agent):
    """
    Base class for agents that need to store an ID or other state.

    Extends the Agent protocol with optional identity tracking.
    Use this when your agent needs to know its own ID.

    Example:
        class MyAgent(StatefulAgent):
            def act(self, observation: Observation) -> Action:
                # Can use self.agent_id if needed
                return {"action": "something"}
    """

    def __init__(self, agent_id: int = -1):
        """
        Initialize with optional agent ID.

        Args:
            agent_id: The agent's node ID in the graph. Default -1 means unassigned.
        """
        self.agent_id = agent_id

    def act(self, observation: Observation) -> Action:
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement act()")

    def __call__(self, observation: Observation, agent_id: int = -1) -> Action:
        """Make agent callable as policy function."""
        return self.act(observation)
