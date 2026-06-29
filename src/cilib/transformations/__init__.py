"""Transformations catalog — atomic, reusable ``GraphState -> GraphState`` steps.

Type function:  ``TransformFactory = Config -> Transform`` (see ``cilib.core.protocols``).
These are the low-level building blocks; a *mechanism* (``cilib.mechanisms``) is a
composition of several. The catalog is the explicit ``REGISTRY`` dict below.
"""
from .message_passing import create_message_passing_transform
from .prediction_market import create_prediction_market_transform
from .token_economy import create_token_budget_calculator
from .updating import belief_update_transform
from .resource import create_resource_transform

# name -> factory ((cfg) -> Transform).
REGISTRY = {
    "message_passing": create_message_passing_transform,
    "prediction_market": create_prediction_market_transform,
    "token_budget": create_token_budget_calculator,
    "belief_update": belief_update_transform,
    "resource": create_resource_transform,
}

__all__ = [
    "create_message_passing_transform",
    "create_prediction_market_transform",
    "create_token_budget_calculator",
    "belief_update_transform",
    "create_resource_transform",
    "REGISTRY",
]
