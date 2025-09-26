# domains/democracy/mechanism_factory.py
from typing import Literal, Dict, Any, Callable
from core.category import Transform, sequential
from core.graph import GraphState
from transformations.bottom_up.message_passing import create_message_passing_transform
from transformations.bottom_up.prediction_market import create_prediction_market_transform
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from transformations.top_down.resource import create_resource_transform
from environments.democracy.configuration import DemocraticEnvironmentConfig

def create_mechanism_pipeline(
    mechanism: Literal["PDD", "PRD", "PLD"],
    config: DemocraticEnvironmentConfig
) -> Transform:
    """
    Create a transformation pipeline for the specified democratic mechanism.
    
    Args:
        mechanism: Type of democratic mechanism
        config: Environment configuration
        
    Returns:
        Composite transformation representing the mechanism
    """
    # Create common components
    message_passing = create_message_passing_transform("communication")
    prediction_market = create_prediction_market_transform(
        config={
            "accuracy": config.prediction_market.accuracy,
            "num_options": len(config.resources.options)
        }
    )
    resource_transform = create_resource_transform(
        config={
            "track_history": config.track_resource_metrics,
            "num_options": len(config.resources.options)
        }
    )
    
    # Mechanism-specific voting aggregation
    if mechanism == "PDD":
        # Direct democracy - all agents vote equally
        voting = create_voting_transform(
            config={"type": "direct"}
        )
        return sequential(
            message_passing,
            prediction_market,
            voting,
            resource_transform
        )
    
    elif mechanism == "PRD":
        # Representative democracy - only representatives vote
        # We can implement this by using a representation flag on agents
        voting = create_voting_transform(
            config={"type": "representative"}
        )
        return sequential(
            message_passing,
            prediction_market,
            voting,
            resource_transform
        )
    
    elif mechanism == "PLD":
        # Liquid democracy - delegation and power flow
        delegation = create_delegation_transform()
        power_flow = create_power_flow_transform()
        voting = create_voting_transform(
            config={"type": "weighted"}
        )
        return sequential(
            message_passing,
            delegation,
            power_flow,
            prediction_market,
            voting,
            resource_transform
        )
    
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")