"""
Agent configuration for farmers_market.

Configures general agents with environment-specific functions:
- Score functions: (observation, action) → float
- Action generators: observation → [actions]
- Action samplers: (observation, key) → action
"""
import jax.numpy as jnp
import jax.random as jr
from typing import List
from engine.agents import RandomAgent, GreedyAgent, Observation, Action


# ==================== Action Generators ====================

def generate_trade_actions(obs: Observation) -> List[Action]:
    """Generate candidate trade actions."""
    actions = [{"trade_offers": {}, "consumption_rate": 0.05}]  # No-op

    neighbors = obs.get("neighbors", [])
    resources = obs.get("resource_types", [])
    my_resources = obs.get("my_resources", {})

    for neighbor in neighbors:
        for resource in resources:
            if my_resources.get(resource, 0) > 10:
                for frac in [0.1, 0.2]:
                    actions.append({
                        "trade_offers": {neighbor: {"resource": resource, "fraction": frac}},
                        "consumption_rate": 0.05
                    })
    return actions


def sample_random_action(obs: Observation, key: jr.PRNGKey) -> Action:
    """Sample random action."""
    neighbors = obs.get("neighbors", [])
    resources = obs.get("resource_types", [])

    trade_offers = {}
    for neighbor in neighbors:
        key, k = jr.split(key)
        if jr.bernoulli(k, p=0.3) and resources:
            key, k1, k2 = jr.split(key, 3)
            resource = resources[int(jr.randint(k1, (), 0, len(resources)))]
            frac = float(jr.uniform(k2, minval=0.05, maxval=0.2))
            trade_offers[neighbor] = {"resource": resource, "fraction": frac}

    key, k = jr.split(key)
    return {
        "trade_offers": trade_offers,
        "consumption_rate": float(jr.uniform(k, minval=0.02, maxval=0.15))
    }


# ==================== Score Functions ====================

def diversity_score(obs: Observation, action: Action) -> float:
    """Score: negative variance of resources (want balance)."""
    resources = obs.get("my_resources", {})
    if not resources:
        return 0.0
    return -float(jnp.var(jnp.array(list(resources.values()))))


def accumulation_score(obs: Observation, action: Action) -> float:
    """Score: total resources (want more)."""
    resources = obs.get("my_resources", {})
    return float(sum(resources.values()))


def trade_benefit_score(obs: Observation, action: Action) -> float:
    """Score: benefit from trading away surplus."""
    resources = obs.get("my_resources", {})
    trade_offers = action.get("trade_offers", {})

    if not trade_offers or not resources:
        return 0.0

    values = jnp.array(list(resources.values()))
    mean = jnp.mean(values)

    score = 0.0
    for _, offer in trade_offers.items():
        resource = offer.get("resource")
        frac = offer.get("fraction", 0)
        if resource in resources:
            amount = resources[resource]
            if amount > mean:
                score += frac * (amount - mean)

    return score


# ==================== Agent Factories ====================

def create_diversity_farmer() -> GreedyAgent:
    """Greedy agent seeking resource balance."""
    return GreedyAgent(diversity_score, generate_trade_actions)


def create_accumulator_farmer() -> GreedyAgent:
    """Greedy agent maximizing total resources."""
    return GreedyAgent(accumulation_score, generate_trade_actions)


def create_trader_farmer() -> GreedyAgent:
    """Greedy agent evaluating trade benefits."""
    return GreedyAgent(trade_benefit_score, generate_trade_actions)


def create_random_farmer(seed: int = 42) -> RandomAgent:
    """Random agent."""
    return RandomAgent(sample_random_action, seed)
