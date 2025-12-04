# core/agents.py
"""
Agent Protocol: Core abstraction for node behaviors.

In Mycorrhiza, an Agent is the fundamental computational unit that processes
observations and produces actions. Agents are nodes in the graph (type 0).

This is a minimal protocol following the functional paradigm:
    Observation → Action

Think of agents like pure functions with optional internal state.
The Agent protocol is used for all node types (Agents, Markets, Democracies)
through the common interface: observe → decide → act.
"""
from typing import Protocol, Dict, Any, TypeAlias

# Type aliases for clarity
Observation: TypeAlias = Dict[str, Any]
Action: TypeAlias = Dict[str, Any]


class Agent(Protocol):
    """
    Protocol for agent behavior.

    Agents are pure functions: Observation → Action.
    They can maintain internal state, but must be callable as functions.

    This protocol is intentionally minimal to enable maximum flexibility:
    - Function-based agents: just a function
    - Class-based agents: implement act() method
    - Stateful agents: can store internal state
    - Stateless agents: pure functions of observation

    Example (Function-based):
        def simple_agent(observation: Observation) -> Action:
            return {"share": observation["resources"] * 0.1}

    Example (Class-based):
        class MyAgent:
            def act(self, observation: Observation) -> Action:
                return {"bid": self.compute_bid(observation)}

    Example (Stateful):
        class LearningAgent:
            def __init__(self):
                self.memory = []

            def act(self, observation: Observation) -> Action:
                self.memory.append(observation)
                return self.make_decision()

    Note: Agent ID is NOT part of the protocol. IDs are managed by the
    graph structure (node index), not by the agent itself.
    """

    def act(self, observation: Observation) -> Action:
        """
        Process observation and produce action.

        Pure function: Given an observation, decide what action to take.

        Args:
            observation: Dictionary containing information the agent perceives.
                        Structure depends on the scenario (market prices,
                        neighbor states, resource levels, etc.)

        Returns:
            Action dictionary specifying the agent's decision.
            Structure depends on the scenario (bids, votes, messages, etc.)
        """
        ...