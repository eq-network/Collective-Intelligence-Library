"""
Prediction Tracking Environment

An environment for the cybernetic dashboard where multiple agents
make predictions, learn from outcomes, and improve calibration over time.

This environment provides:
- Multi-agent initialization
- Prediction/resolution pipeline
- Agent personality presets (optimist, realist, pessimist)
- Helper functions for common scenarios
"""

import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional
from core.time import World, Event, EventLog, tick_world
from core.graph import GraphState


# =============================================================================
# CONSTANTS
# =============================================================================

USER_TYPE = 0
AGENT_TYPE = 1
ESTIMATOR_TYPE = 2

# Agent personality presets - starting calibration biases
PERSONALITIES = {
    "optimist": {
        "initial_bias": 0.15,      # Tends to be 15% overconfident
        "learning_rate": 0.3       # Adjusts quickly
    },
    "realist": {
        "initial_bias": 0.0,       # Well-calibrated from start
        "learning_rate": 0.1       # Maintains calibration
    },
    "pessimist": {
        "initial_bias": -0.15,     # Tends to be 15% underconfident
        "learning_rate": 0.3       # Adjusts quickly
    },
    "stubborn": {
        "initial_bias": 0.2,       # Very overconfident
        "learning_rate": 0.05      # Learns slowly
    }
}


# =============================================================================
# ENVIRONMENT INITIALIZATION
# =============================================================================

def create_prediction_environment(
    agent_names: List[str],
    agent_personalities: Optional[List[str]] = None
) -> World:
    """
    Initialize a prediction tracking environment.

    Args:
        agent_names: List of agent names (e.g., ["alice", "bob", "charlie"])
        agent_personalities: Optional list of personality types for each agent
                           If None, all agents are "realist"

    Returns:
        World with user (node 0) + agents (nodes 1, 2, ...)
    """
    n_agents = len(agent_names) + 1  # +1 for user

    if agent_personalities is None:
        agent_personalities = ["realist"] * len(agent_names)

    # Node types: user (0), then agents (1, 1, 1, ...)
    node_types = jnp.array([USER_TYPE] + [AGENT_TYPE] * len(agent_names), dtype=jnp.int32)

    # Store agent metadata in global_attrs
    agent_metadata = {
        "user": {"type": "user", "name": "user"}
    }
    for i, (name, personality) in enumerate(zip(agent_names, agent_personalities), start=1):
        agent_metadata[str(i)] = {
            "name": name,
            "personality": personality,
            **PERSONALITIES.get(personality, PERSONALITIES["realist"])
        }

    state = GraphState(
        node_types=node_types,
        node_attrs={
            "calibration": jnp.full(n_agents, -1.0),  # -1.0 = no calibration yet
            "prediction_count": jnp.zeros(n_agents, dtype=jnp.int32),
            "task_count": jnp.zeros(n_agents, dtype=jnp.int32),
            "success_count": jnp.zeros(n_agents, dtype=jnp.int32),
        },
        adj_matrices={
            "trust": jnp.zeros((n_agents, n_agents)),  # Trust relationships
        },
        edge_attrs={},
        global_attrs={
            "tick": 0,
            "agents": agent_metadata
        }
    )

    return World(
        tick=0,
        state=state,
        log=EventLog(completed=[], pending=[])
    )


# =============================================================================
# AGENT BEHAVIOR: Personality-based prediction adjustment
# =============================================================================

def adjust_probability_for_personality(
    base_probability: float,
    personality: str,
    current_calibration: float
) -> float:
    """
    Adjust a base probability based on agent personality and current calibration.

    This simulates agents with different prediction biases that learn over time.

    Args:
        base_probability: The "true" probability if perfectly calibrated
        personality: Agent personality type
        current_calibration: Agent's current calibration score (-1.0 if none)

    Returns:
        Adjusted probability reflecting personality bias
    """
    if personality not in PERSONALITIES:
        return base_probability

    config = PERSONALITIES[personality]
    initial_bias = config["initial_bias"]
    learning_rate = config["learning_rate"]

    # If no calibration yet, use initial bias
    if current_calibration < 0:
        adjusted = base_probability + initial_bias
    else:
        # Learn from past calibration
        # If calibration is high (bad), reduce confidence
        # If calibration is low (good), maintain confidence
        correction = -current_calibration * learning_rate
        adjusted = base_probability + initial_bias + correction

    # Clamp to [0.05, 0.95] - agents don't predict certainties
    return max(0.05, min(0.95, adjusted))


def get_agent_personality(world: World, agent_id: int) -> str:
    """Get an agent's personality type from world metadata."""
    agent_meta = world.state.global_attrs["agents"].get(str(agent_id))
    if agent_meta:
        return agent_meta.get("personality", "realist")
    return "realist"


# =============================================================================
# SCENARIO GENERATORS
# =============================================================================

def generate_task_scenario(
    world: World,
    agent_id: int,
    true_completion_probability: float,
    horizon: int,
    task_id: Optional[str] = None
) -> Tuple[World, float, bool]:
    """
    Generate a task scenario with personality-adjusted prediction.

    Args:
        world: Current world state
        agent_id: Which agent is working on the task
        true_completion_probability: The actual probability task completes
        horizon: When the prediction should resolve
        task_id: Optional task identifier

    Returns:
        (world_with_prediction, adjusted_probability, will_complete)
    """
    if task_id is None:
        task_id = f"task_{world.tick}_{agent_id}"

    # Get agent's personality
    personality = get_agent_personality(world, agent_id)
    current_calibration = float(world.state.node_attrs["calibration"][agent_id])

    # Agent predicts based on personality
    predicted_probability = adjust_probability_for_personality(
        true_completion_probability,
        personality,
        current_calibration
    )

    # Determine if task actually completes (stochastic)
    import random
    will_complete = random.random() < true_completion_probability

    # Register prediction
    event = Event(
        tick=world.tick,
        source=agent_id,
        target=agent_id,
        event_type="prediction",
        payload={
            "condition": "task_complete",
            "probability": predicted_probability,
            "horizon": horizon,
            "prediction_id": f"pred_{task_id}",
            "task_id": task_id,
            "true_probability": true_completion_probability  # For analysis
        },
        duration=0
    )

    new_log = world.log.append(event)
    new_world = World(tick=world.tick, state=world.state, log=new_log)

    return new_world, predicted_probability, will_complete


def complete_task_at_tick(
    world: World,
    agent_id: int,
    completion_tick: int,
    task_id: str
) -> World:
    """
    Schedule a task completion at a specific tick.

    This doesn't execute immediately - it creates an event that will
    complete at the specified tick.
    """
    duration = completion_tick - world.tick

    event = Event(
        tick=world.tick,
        source=agent_id,
        target=agent_id,
        event_type="task_complete",
        payload={"task_id": task_id},
        duration=max(0, duration)
    )

    # Update task counts immediately
    new_task_count = world.state.node_attrs["task_count"].at[agent_id].add(1)
    new_success_count = world.state.node_attrs["success_count"].at[agent_id].add(1)

    new_state = world.state.replace(
        node_attrs={
            **world.state.node_attrs,
            "task_count": new_task_count,
            "success_count": new_success_count
        }
    )

    new_log = world.log.append(event)
    return World(tick=world.tick, state=new_state, log=new_log)


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def get_agent_stats(world: World, agent_id: int) -> Dict:
    """
    Get comprehensive stats for an agent.

    Returns dict with:
    - name
    - personality
    - calibration
    - prediction_count
    - task_count
    - success_rate
    """
    agent_meta = world.state.global_attrs["agents"].get(str(agent_id), {})

    calibration = float(world.state.node_attrs["calibration"][agent_id])
    prediction_count = int(world.state.node_attrs["prediction_count"][agent_id])
    task_count = int(world.state.node_attrs["task_count"][agent_id])
    success_count = int(world.state.node_attrs["success_count"][agent_id])

    success_rate = success_count / task_count if task_count > 0 else 0.0

    return {
        "agent_id": agent_id,
        "name": agent_meta.get("name", f"agent_{agent_id}"),
        "personality": agent_meta.get("personality", "unknown"),
        "calibration": calibration if calibration >= 0 else None,
        "prediction_count": prediction_count,
        "task_count": task_count,
        "success_count": success_count,
        "success_rate": success_rate
    }


def get_leaderboard(world: World) -> List[Dict]:
    """
    Get agent leaderboard sorted by calibration (best first).

    Returns list of agent stats, sorted by calibration (lower is better).
    """
    n_agents = len(world.state.node_types)
    stats = []

    for agent_id in range(1, n_agents):  # Skip user (0)
        agent_stats = get_agent_stats(world, agent_id)
        if agent_stats["calibration"] is not None:
            stats.append(agent_stats)

    # Sort by calibration (lower is better)
    return sorted(stats, key=lambda x: x["calibration"])


def compute_correction_strength(world: World, agent_id: int) -> Optional[float]:
    """
    Compute correction strength: does agent learn from errors?

    High correction strength (>0.5) means agent adjusts predictions
    based on past performance.

    Returns:
        float in [0, 1] or None if insufficient data
    """
    # Get agent's prediction history
    predictions = [
        e for e in world.log.completed
        if e.event_type == "prediction" and e.source == agent_id
    ]

    resolutions = [
        e for e in world.log.completed
        if e.event_type == "resolution" and e.source == agent_id
    ]

    if len(resolutions) < 3:
        return None  # Need at least 3 to measure

    # Compute: correlation between past errors and future adjustments
    # Simplified: just check if calibration is improving
    brier_scores = [r.payload["brier_score"] for r in resolutions]

    if len(brier_scores) < 2:
        return None

    # Simple metric: is most recent better than average of earlier ones?
    early_avg = sum(brier_scores[:-1]) / len(brier_scores[:-1])
    recent = brier_scores[-1]

    if recent < early_avg * 0.8:  # 20% improvement
        return 0.8
    elif recent < early_avg:
        return 0.5
    else:
        return 0.2


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_agent_stats(world: World, agent_id: int):
    """Pretty print an agent's statistics."""
    stats = get_agent_stats(world, agent_id)

    print(f"\nAgent: {stats['name']} (ID: {agent_id})")
    print(f"  Personality: {stats['personality']}")
    print(f"  Calibration: {stats['calibration']:.4f}" if stats['calibration'] else "  Calibration: None (no predictions yet)")
    print(f"  Predictions: {stats['prediction_count']}")
    print(f"  Tasks: {stats['task_count']} ({stats['success_rate']*100:.0f}% success)")


def print_leaderboard(world: World):
    """Pretty print the agent leaderboard."""
    leaderboard = get_leaderboard(world)

    print("\n" + "="*60)
    print("AGENT LEADERBOARD (by calibration)")
    print("="*60)

    for i, stats in enumerate(leaderboard, 1):
        medal = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}.get(i, f"[{i}th]")
        print(f"\n{medal} {stats['name']}")
        print(f"    Calibration: {stats['calibration']:.4f}")
        print(f"    Personality: {stats['personality']}")
        print(f"    Predictions: {stats['prediction_count']}")
        print(f"    Success Rate: {stats['success_rate']*100:.0f}%")

    print("\n" + "="*60)
