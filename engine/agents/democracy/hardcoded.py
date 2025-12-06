# engine/agents/democracy/hardcoded.py
"""
Defines agent classes with deterministic, hardcoded logic.
These are useful for baseline testing and creating predictable scenarios.
"""
from typing import Dict, Any, Optional
import numpy as np

from engine.agents.base import StatefulAgent, Action
# We add Optional[LLMService] to the init signature for interface consistency, even if not used.
from services.llm import LLMService


class HardcodedAlignedAgent(StatefulAgent):
    """
    An agent that is perfectly aligned and acts based on deterministic rules.
    It has access to perfect information and always chooses the highest-yield portfolio.
    """
    def __init__(self, agent_id: int, role: str, mechanism: str, num_portfolios: int, llm_service: Optional[LLMService] = None):
        super().__init__(agent_id)
        self.role = role
        # This agent does not use the llm_service, but accepts it for a consistent interface.

    def act(self, observation: Dict[str, Any]) -> Action:
        """Chooses the single best portfolio based on the provided yields."""
        portfolio_yields = observation.get("portfolio_yields", [])
        if not portfolio_yields:
            return {"type": "vote", "votes": []}

        best_portfolio_idx = np.argmax(portfolio_yields)
        votes = [0] * len(portfolio_yields)
        votes[best_portfolio_idx] = 1
        
        return {"type": "vote", "votes": votes}

class HardcodedAdversarialAgent(StatefulAgent):
    """
    An agent that is perfectly adversarial and acts based on deterministic rules.
    It has access to perfect information and always chooses the lowest-yield portfolio.
    """
    def __init__(self, agent_id: int, role: str, mechanism: str, num_portfolios: int, llm_service: Optional[LLMService] = None):
        super().__init__(agent_id)
        self.role = role

    def act(self, observation: Dict[str, Any]) -> Action:
        """Chooses the single worst portfolio based on the provided yields."""
        portfolio_yields = observation.get("portfolio_yields", [])
        if not portfolio_yields:
            return {"type": "vote", "votes": []}

        worst_portfolio_idx = np.argmin(portfolio_yields)
        votes = [0] * len(portfolio_yields)
        votes[worst_portfolio_idx] = 1

        return {"type": "vote", "votes": votes}