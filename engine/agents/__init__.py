"""
General agents: domain-agnostic policies.

Agent: Observation → Action
"""
from engine.agents.base import Agent, Observation, Action, AgentPolicy
from engine.agents.random_agent import RandomAgent
from engine.agents.greedy_agent import GreedyAgent
from engine.agents.llm_agent import LLMAgent

__all__ = [
    'Agent',
    'Observation',
    'Action',
    'AgentPolicy',
    'RandomAgent',
    'GreedyAgent',
    'LLMAgent',
]
