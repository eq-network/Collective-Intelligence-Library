# core/environment.py
"""
Defines the abstract base class for a simulation environment.

Supports two modes:
1. Transform-based: a composed Transform (GraphState -> GraphState) drives each step
2. Agent-based (legacy): obs -> act -> apply cycle

Subclasses using transforms only need to implement is_terminated() and reset().
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from core.agents import Agent, Action
from core.graph import GraphState
from core.category import Transform

logger = logging.getLogger(__name__)


class Environment(ABC):
    """
    Abstract Base Class for a simulation environment.

    If step_transform is provided, step() applies it directly to GraphState.
    Otherwise, falls back to the agent-based obs -> act -> apply cycle.
    """
    def __init__(self, initial_state: GraphState, agents: List[Agent] = None,
                 step_transform: Transform = None):
        self.state = initial_state
        self._initial_state = initial_state
        self.agents = agents or []
        self._step_transform = step_transform
        self.round_num = 0
        self.history = []

    def get_observation_for_agent(self, agent: Agent) -> Dict[str, Any]:
        raise NotImplementedError("Implement for agent-based stepping")

    def apply_actions(self, actions: List[Action]) -> GraphState:
        raise NotImplementedError("Implement for agent-based stepping")

    def step(self) -> Tuple[GraphState, bool]:
        if self._step_transform is not None:
            self.state = self._step_transform(self.state)
        else:
            observations = {
                agent.agent_id: self.get_observation_for_agent(agent)
                for agent in self.agents
            }
            actions = [agent.act(observations[agent.agent_id]) for agent in self.agents]
            self.state = self.apply_actions(actions)

        self.round_num += 1
        self.history.append(self.state)
        return self.state, self.is_terminated()

    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    def run(self, num_rounds: int) -> List[GraphState]:
        self.reset()
        for i in range(num_rounds):
            logger.debug(f"Round {i+1}/{num_rounds}")
            _, terminated = self.step()
            if terminated:
                logger.info(f"Terminated at round {i+1}")
                break
        return self.history

    def reset(self) -> None:
        self.state = self._initial_state
        self.round_num = 0
        self.history = []
