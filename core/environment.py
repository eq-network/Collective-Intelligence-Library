# core/environment.py
"""
Defines the abstract base class for a simulation environment.

The environment manages the global state, orchestrates the simulation loop,
and provides observations to agents.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from core.agents import Agent, Action
from core.graph import GraphState

class Environment(ABC):
    """
    Abstract Base Class for a simulation environment.

    The environment is responsible for:
    - Maintaining the global state of the simulation (GraphState).
    - Managing a population of agents.
    - Running the simulation loop step-by-step.
    - Providing agents with observations.
    - Applying the outcomes of agent actions to the state.
    """
    def __init__(self, agents: List[Agent], initial_state: GraphState):
        self.agents = agents
        self.state = initial_state
        self.round_num = 0
        self.history = []

    @abstractmethod
    def get_observation_for_agent(self, agent: Agent) -> Dict[str, Any]:
        """
        Constructs the observation dictionary for a specific agent.
        This method determines what an agent can perceive from the world.
        """
        pass

    @abstractmethod
    def apply_actions(self, actions: List[Action]) -> GraphState:
        """
        Processes a list of agent actions and applies their collective
        effect to the environment's state by executing a world update transform.
        """
        pass

    def step(self) -> Tuple[GraphState, bool]:
        """
        Executes a single step (round) of the simulation.

        Returns:
            A tuple containing the new state and a boolean indicating
            if the simulation has terminated.
        """
        # 1. Get observations for all agents
        observations = {agent.agent_id: self.get_observation_for_agent(agent) for agent in self.agents}

        # 2. Collect actions from all agents based on their observations
        actions = [agent.act(observations[agent.agent_id]) for agent in self.agents]

        # 3. Apply the collective actions to update the environment state
        new_state = self.apply_actions(actions)
        self.state = new_state

        # 4. Update internal tracking and check for termination
        self.round_num += 1
        self.history.append(new_state)
        terminated = self.is_terminated()

        return self.state, terminated

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Checks if the simulation should terminate based on the current state.
        (e.g., resources below a threshold, max rounds reached).
        """
        pass

    def run(self, num_rounds: int) -> List[GraphState]:
        """
        Runs the simulation for a specified number of rounds.
        """
        print(f"Starting simulation run for {num_rounds} rounds...")
        self.reset()
        for i in range(num_rounds):
            print(f"\n--- Round {i+1}/{num_rounds} ---")
            _, terminated = self.step()
            if terminated:
                print(f"Simulation terminated early at round {i+1}.")
                break
        print("\nSimulation run finished.")
        return self.history

    @abstractmethod
    def reset(self) -> None:
        """Resets the environment to its initial state."""
        pass