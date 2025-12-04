# core/environment.py
"""
Environment Protocol: Core abstraction for scenario factories.

In Mycorrhiza, an "Environment" is not a stateful container (like OpenAI Gym).
Instead, it's a SCENARIO FACTORY - a way to create initial GraphState configurations.

Think of environments as scenarios:
- FarmersMarket: Creates a market-based trading scenario
- TragedyOfCommons: Creates a shared resource dilemma
- PollutionGame: Creates an environmental game scenario

The environment is just the initial GraphState configuration.
Everything else is pure transforms.
"""
from typing import Protocol, Dict, Any
from core.graph import GraphState


class Environment(Protocol):
    """
    Protocol for scenario factories.

    An Environment creates initial GraphState configurations for different
    scenarios (like CartPole, MountainCar in RL, but for collective intelligence).

    This is a minimal protocol - environments are just factories for initial state.
    All simulation logic happens through pure Transform functions.

    Example:
        class FarmersMarket:
            def create_initial_state(self, num_farmers: int, seed: int) -> GraphState:
                # Setup farmers market scenario
                return GraphState(...)

        # Usage
        env = FarmersMarket()
        state = env.create_initial_state(num_farmers=10, seed=42)

        # Run with transforms (no environment needed after initialization)
        transform = create_market_transform()
        for _ in range(100):
            state = transform(state)
    """

    def create_initial_state(self, **params) -> GraphState:
        """
        Create initial graph state for this scenario.

        Args:
            **params: Scenario-specific parameters (num_agents, resources, etc.)

        Returns:
            Initial GraphState configured for this scenario
        """
        ...