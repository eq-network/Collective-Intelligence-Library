"""
General agents: domain-agnostic policies.

Agent: Observation → Action

The canonical Agent protocol is defined in core/agents.py.
This module re-exports it along with StatefulAgent for agents needing identity.
"""
from engine.agents.base import Agent, Observation, Action, AgentPolicy, StatefulAgent
from engine.agents.random_agent import RandomAgent
from engine.agents.greedy_agent import GreedyAgent
from engine.agents.llm_agent import LLMAgent

__all__ = [
    'Agent',
    'StatefulAgent',
    'Observation',
    'Action',
    'AgentPolicy',
    'RandomAgent',
    'GreedyAgent',
    'LLMAgent',
]
