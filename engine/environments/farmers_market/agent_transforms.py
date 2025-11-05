"""
Agent-driven transformations for Farmer's Market.

Transforms = infrastructure (message passing)
Agents = policy (decisions)
"""
import jax.numpy as jnp
from typing import List, Dict, Any, Callable, Optional
from core.graph import GraphState
from core.category import Transform, sequential

from engine.environments.farmers_market.observations import build_all_observations
from engine.agents import Agent


# ==================== Helper Transforms ====================

def create_resource_growth_transform() -> Transform:
    """Grow resources by growth rates."""
    def transform(state: GraphState) -> GraphState:
        resources = state.global_attrs.get("resource_types", [])
        new_node_attrs = dict(state.node_attrs)

        for rt in resources:
            rkey = f"resources_{rt}"
            gkey = f"growth_rate_{rt}"
            if rkey in state.node_attrs and gkey in state.node_attrs:
                new_node_attrs[rkey] = state.node_attrs[rkey] * state.node_attrs[gkey]

        return state.replace(node_attrs=new_node_attrs)
    return transform


def create_round_counter_transform() -> Transform:
    """Increment round counter."""
    def transform(state: GraphState) -> GraphState:
        new_global = dict(state.global_attrs)
        new_global["round"] = state.global_attrs.get("round", 0) + 1
        return state.replace(global_attrs=new_global)
    return transform


def create_history_tracker_transform() -> Transform:
    """Track state in history."""
    def transform(state: GraphState) -> GraphState:
        resources = state.global_attrs.get("resource_types", [])
        totals = {rt: float(jnp.sum(state.node_attrs[f"resources_{rt}"])) for rt in resources}

        entry = {
            "round": state.global_attrs.get("round", 0),
            "total_trades": state.global_attrs.get("total_trades", 0),
            "total_resources": totals
        }

        new_global = dict(state.global_attrs)
        history = new_global.get("history", []) + [entry]
        new_global["history"] = history

        return state.replace(global_attrs=new_global)
    return transform


# ==================== Agent-Driven Transforms ====================

def create_agent_driven_trade_transform(
    agent_policies: List[Agent]
) -> Transform:
    """
    Creates a trade transformation driven by agent decisions.

    The transform is pure infrastructure:
    1. Builds observations for each agent
    2. Calls agent policy functions to get trade decisions
    3. Routes trade messages between agents
    4. Updates state based on executed trades

    Args:
        agent_policies: List of agent policy functions, one per agent

    Returns:
        A pure transformation function

    Properties preserved:
        - Resource conservation (trades are symmetric exchanges)
        - Network topology unchanged
    """
    def transform(state: GraphState) -> GraphState:
        num_agents = state.num_nodes

        # Validate agent count
        if len(agent_policies) != num_agents:
            raise ValueError(
                f"Number of agents ({len(agent_policies)}) doesn't match "
                f"number of nodes ({num_agents})"
            )

        # Step 1: Build observations for all agents
        observations = build_all_observations(state)

        # Step 2: Query each agent for their action
        actions = []
        for agent_id, (obs, policy) in enumerate(zip(observations, agent_policies)):
            action = policy(obs, agent_id)
            actions.append(action)

        # Step 3: Process trade offers into actual trades
        resource_types = state.global_attrs.get("resource_types", [])
        new_node_attrs = dict(state.node_attrs)

        # Build trade pairs from offers
        trades_executed = 0

        for agent_id, action in enumerate(actions):
            trade_offers = action.get("trade_offers", {})

            for partner_id, offer in trade_offers.items():
                # Check if this is a mutual trade (both offered to each other)
                partner_action = actions[partner_id]
                partner_offers = partner_action.get("trade_offers", {})

                # Only execute if it's a valid trade
                if partner_id < agent_id:
                    # Already processed this pair
                    continue

                # Execute the trade based on offers
                resource_to_trade = offer.get("resource")
                offer_fraction = offer.get("fraction", 0.0)

                if resource_to_trade and offer_fraction > 0:
                    resource_key = f"resources_{resource_to_trade}"

                    if resource_key in state.node_attrs:
                        current_resources = new_node_attrs[resource_key].copy()

                        # Agent i offers to give fraction of their resource
                        from_agent_i = current_resources[agent_id] * offer_fraction

                        # Check if partner also made an offer
                        if agent_id in partner_offers:
                            partner_offer = partner_offers[agent_id]
                            partner_resource = partner_offer.get("resource")
                            partner_fraction = partner_offer.get("fraction", 0.0)

                            # If both offering same resource, do symmetric trade
                            if partner_resource == resource_to_trade and partner_fraction > 0:
                                from_partner = current_resources[partner_id] * partner_fraction

                                # Symmetric exchange
                                current_resources = current_resources.at[agent_id].add(-from_agent_i + from_partner)
                                current_resources = current_resources.at[partner_id].add(-from_partner + from_agent_i)

                                new_node_attrs[resource_key] = current_resources
                                trades_executed += 1
                        else:
                            # One-sided offer: agent gives, partner accepts
                            # For now, do a simple symmetric fractional trade
                            from_partner = current_resources[partner_id] * offer_fraction

                            current_resources = current_resources.at[agent_id].add(-from_agent_i + from_partner)
                            current_resources = current_resources.at[partner_id].add(-from_partner + from_agent_i)

                            new_node_attrs[resource_key] = current_resources
                            trades_executed += 1

        # Update global attributes
        new_global_attrs = dict(state.global_attrs)
        new_global_attrs["total_trades"] = state.global_attrs.get("total_trades", 0) + trades_executed

        return state.replace(
            node_attrs=new_node_attrs,
            global_attrs=new_global_attrs
        )

    return transform


def create_agent_driven_consumption_transform(
    agent_policies: List[Agent]
) -> Transform:
    """
    Creates a consumption transformation driven by agent decisions.

    Each agent decides their own consumption rate based on their policy.

    Args:
        agent_policies: List of agent policy functions

    Returns:
        A pure transformation function
    """
    def transform(state: GraphState) -> GraphState:
        num_agents = state.num_nodes

        if len(agent_policies) != num_agents:
            raise ValueError(f"Agent count mismatch: {len(agent_policies)} vs {num_agents}")

        # Build observations
        observations = build_all_observations(state)

        # Query agents for consumption decisions
        resource_types = state.global_attrs.get("resource_types", [])
        new_node_attrs = dict(state.node_attrs)

        for agent_id, (obs, policy) in enumerate(zip(observations, agent_policies)):
            action = policy(obs, agent_id)
            consumption_rate = action.get("consumption_rate", 0.05)

            # Apply consumption to each resource
            for resource_type in resource_types:
                resource_key = f"resources_{resource_type}"

                if resource_key in state.node_attrs:
                    current_resources = new_node_attrs[resource_key].copy()
                    current_amount = current_resources[agent_id]

                    # Consume the specified fraction
                    consumed = current_amount * consumption_rate
                    new_amount = max(0.0, current_amount - consumed)

                    current_resources = current_resources.at[agent_id].set(new_amount)
                    new_node_attrs[resource_key] = current_resources

        return state.replace(node_attrs=new_node_attrs)

    return transform


def create_agent_driven_round_transform(
    agent_policies: List[Agent],
    include_growth: bool = True,
    include_trade: bool = True,
    include_consumption: bool = True
) -> Transform:
    """
    Creates a complete round transformation driven by agents.

    Composes growth, trade, and consumption into one round, where
    trade and consumption are determined by agent policies.

    Args:
        agent_policies: List of agent policy functions
        include_growth: Whether to include resource growth
        include_trade: Whether to include trading
        include_consumption: Whether to include consumption

    Returns:
        A composed transformation representing one complete round
    """
    transforms = []

    if include_growth:
        transforms.append(create_resource_growth_transform())

    if include_trade:
        transforms.append(create_agent_driven_trade_transform(agent_policies))

    if include_consumption:
        transforms.append(create_agent_driven_consumption_transform(agent_policies))

    # Always track history and increment round
    transforms.append(create_history_tracker_transform())
    transforms.append(create_round_counter_transform())

    return sequential(*transforms)
