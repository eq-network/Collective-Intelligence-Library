"""
LLM agent.

Observation → Prompt → LLM → Response → Action
"""
from typing import Callable
from engine.agents.base import Agent, Observation, Action


class LLMAgent(Agent):
    """LLM policy: observation → prompt → LLM → action."""

    def __init__(
        self,
        obs_to_prompt: Callable[[Observation], str],
        response_to_action: Callable[[str, Observation], Action],
        llm_fn: Callable[[str], str]
    ):
        self.obs_to_prompt = obs_to_prompt
        self.response_to_action = response_to_action
        self.llm_fn = llm_fn

    def act(self, observation: Observation) -> Action:
        prompt = self.obs_to_prompt(observation)
        response = self.llm_fn(prompt)
        return self.response_to_action(response, observation)
