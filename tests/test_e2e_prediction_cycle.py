"""
End-to-End Test 1: The Core Prediction Cycle

Tests the fundamental cybernetic loop:
  Prediction -> Outcome -> Resolution -> Brier Score -> Calibration Update

Uses a pipeline model where each tick applies a series of transformations
to the GraphState.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
from core.time import World, Event, EventLog, tick_world
from core.graph import GraphState


# =============================================================================
# FIXTURES: Graph Initialization
# =============================================================================

def create_initial_state(n_agents: int = 2) -> GraphState:
    """
    Create initial GraphState with user (node 0) + agents.

    Pipeline style: This is a pure function that constructs state.
    """
    # Node types: 0 = USER, 1 = AGENT
    node_types = jnp.array([0] + [1] * (n_agents - 1), dtype=jnp.int32)

    return GraphState(
        node_types=node_types,
        node_attrs={
            "calibration": jnp.full(n_agents, -1.0),  # -1.0 = no calibration yet
            "prediction_count": jnp.zeros(n_agents, dtype=jnp.int32),
            "task_count": jnp.zeros(n_agents, dtype=jnp.int32),
        },
        adj_matrices={},
        edge_attrs={},
        global_attrs={"tick": 0}
    )


def create_world(n_agents: int = 2) -> World:
    """Initialize a world with user + agents."""
    return World(
        tick=0,
        state=create_initial_state(n_agents),
        log=EventLog(completed=[], pending=[])
    )


# =============================================================================
# PIPELINE FUNCTIONS: Each transforms World -> World
# =============================================================================

def register_prediction(
    world: World,
    agent_id: int,
    probability: float,
    horizon: int,
    condition: str = "task_complete"
) -> World:
    """
    Pipeline function: Register a prediction event.

    World -> World

    Predictions are instant (duration=0), so they complete immediately.
    """
    event = Event(
        tick=world.tick,
        source=agent_id,
        target=agent_id,
        event_type="prediction",
        payload={
            "condition": condition,
            "probability": probability,
            "horizon": horizon,
            "prediction_id": f"pred_{world.tick}_{agent_id}"
        },
        duration=0  # Instant
    )

    new_log = world.log.append(event)
    return World(tick=world.tick, state=world.state, log=new_log)


def mark_task_complete(world: World, agent_id: int, task_id: str = "default") -> World:
    """
    Pipeline function: Mark a task as complete.

    World -> World
    """
    event = Event(
        tick=world.tick,
        source=agent_id,
        target=agent_id,
        event_type="task_complete",
        payload={"task_id": task_id},
        duration=0  # Instant
    )

    # Update task count in state
    new_task_count = world.state.node_attrs["task_count"].at[agent_id].add(1)
    new_state = world.state.replace(
        node_attrs={**world.state.node_attrs, "task_count": new_task_count}
    )

    new_log = world.log.append(event)
    return World(tick=world.tick, state=new_state, log=new_log)


def check_and_resolve_predictions(world: World) -> World:
    """
    Pipeline function: Check if any predictions should resolve at current tick.

    World -> World

    For each prediction where horizon == current_tick:
    1. Check if the condition is met
    2. Compute Brier score
    3. Create resolution event
    4. Update agent calibration
    """
    current_tick = world.tick

    # Find predictions that resolve at this tick
    predictions = [
        e for e in world.log.completed
        if e.event_type == "prediction"
        and e.payload["horizon"] == current_tick
    ]

    if not predictions:
        return world  # No predictions to resolve

    new_state = world.state
    new_log = world.log

    for pred in predictions:
        # Check if already resolved
        pred_id = pred.payload["prediction_id"]
        already_resolved = any(
            e.event_type == "resolution" and e.payload.get("prediction_id") == pred_id
            for e in new_log.completed
        )
        if already_resolved:
            continue

        # Check condition
        condition = pred.payload["condition"]
        agent_id = pred.source

        # Look for task completion events
        task_complete = any(
            e.event_type == "task_complete"
            and e.source == agent_id
            and e.tick <= current_tick
            for e in new_log.completed
        )

        outcome = task_complete
        probability = pred.payload["probability"]

        # Brier score: (p - outcome)^2
        brier = (probability - float(outcome)) ** 2

        # Create resolution event
        resolution = Event(
            tick=current_tick,
            source=agent_id,
            target=agent_id,
            event_type="resolution",
            payload={
                "prediction_id": pred_id,
                "outcome": outcome,
                "brier_score": brier
            },
            duration=0
        )

        new_log = new_log.append(resolution)

        # Update calibration: running average
        current_calibration = float(new_state.node_attrs["calibration"][agent_id])
        current_count = int(new_state.node_attrs["prediction_count"][agent_id])

        if current_calibration < 0:  # -1.0 = no calibration yet
            # First prediction
            new_calibration = brier
        else:
            # Running average
            new_calibration = (current_calibration * current_count + brier) / (current_count + 1)

        # Update state (using JAX immutable arrays)
        new_calibration_array = new_state.node_attrs["calibration"].at[agent_id].set(new_calibration)
        new_count = new_state.node_attrs["prediction_count"].at[agent_id].add(1)

        new_state = new_state.replace(
            node_attrs={
                **new_state.node_attrs,
                "calibration": new_calibration_array,
                "prediction_count": new_count
            }
        )

    return World(tick=current_tick, state=new_state, log=new_log)


def tick_pipeline(world: World) -> World:
    """
    The tick pipeline: a composition of transformations.

    World -> World

    Each tick applies:
    1. tick_world (advance time, complete pending events)
    2. check_and_resolve_predictions (resolve any predictions at this tick)
    """
    # First: advance time and complete pending events
    world_after_tick = tick_world(world)

    # Second: check for prediction resolutions
    world_after_resolution = check_and_resolve_predictions(world_after_tick)

    return world_after_resolution


def run_n_ticks(world: World, n: int) -> World:
    """
    Run N iterations of the tick pipeline.

    This is a fold/reduce operation: apply tick_pipeline N times.
    """
    current = world
    for _ in range(n):
        current = tick_pipeline(current)
    return current


# =============================================================================
# TEST: The Core Prediction Cycle
# =============================================================================

def test_prediction_cycle_success():
    """
    Test Case 1: Prediction comes true

    Flow:
    - Tick 0: Agent predicts task will complete by tick 10, p=0.8
    - Tick 5: Task actually completes
    - Tick 10: Prediction resolves -> outcome=True
    - Brier score = (0.8 - 1.0)^2 = 0.04
    - Calibration updates to 0.04
    """
    # Setup
    world = create_world(n_agents=2)  # User + 1 agent

    print("\n=== Tick 0: Register Prediction ===")
    world = register_prediction(
        world,
        agent_id=1,
        probability=0.8,
        horizon=10,
        condition="task_complete"
    )

    # Verify prediction registered
    predictions = [e for e in world.log.completed if e.event_type == "prediction"]
    assert len(predictions) == 1
    assert predictions[0].payload["probability"] == 0.8
    assert predictions[0].payload["horizon"] == 10

    print(f"  [OK] Prediction registered: p=0.8, horizon=10")
    print(f"  Current tick: {world.tick}")

    # Advance to tick 5
    print("\n=== Tick 1-5: Time passes ===")
    world = run_n_ticks(world, 5)
    assert world.tick == 5
    print(f"  [OK] Advanced to tick {world.tick}")

    # Task completes
    print("\n=== Tick 5: Task Completes ===")
    world = mark_task_complete(world, agent_id=1, task_id="test_task")

    task_events = [e for e in world.log.completed if e.event_type == "task_complete"]
    assert len(task_events) == 1
    assert world.state.node_attrs["task_count"][1] == 1
    print(f"  [OK] Task completed at tick {world.tick}")

    # Advance to tick 10 (prediction horizon)
    print("\n=== Tick 6-10: Time passes to horizon ===")
    world = run_n_ticks(world, 5)
    assert world.tick == 10
    print(f"  [OK] Reached horizon at tick {world.tick}")

    # Verify prediction resolved
    print("\n=== Tick 10: Prediction Resolves ===")
    resolution_events = [e for e in world.log.completed if e.event_type == "resolution"]
    assert len(resolution_events) == 1

    resolution = resolution_events[0]
    assert resolution.payload["outcome"] == True
    assert abs(resolution.payload["brier_score"] - 0.04) < 0.001  # (0.8 - 1.0)^2
    print(f"  [OK] Resolution: outcome={resolution.payload['outcome']}")
    print(f"  [OK] Brier score: {resolution.payload['brier_score']:.4f}")

    # Verify calibration updated
    calibration = world.state.node_attrs["calibration"][1]
    prediction_count = world.state.node_attrs["prediction_count"][1]

    assert abs(calibration - 0.04) < 0.001
    assert prediction_count == 1
    print(f"  [OK] Agent calibration: {calibration:.4f}")
    print(f"  [OK] Prediction count: {prediction_count}")


def test_prediction_cycle_failure():
    """
    Test Case 2: Prediction fails (task doesn't complete)

    Flow:
    - Tick 0: Agent predicts task will complete by tick 10, p=0.9
    - Tick 10: Task has NOT completed
    - Tick 10: Prediction resolves -> outcome=False
    - Brier score = (0.9 - 0.0)^2 = 0.81 (bad prediction!)
    - Calibration updates to 0.81
    """
    world = create_world(n_agents=2)

    print("\n=== Tick 0: Register Prediction ===")
    world = register_prediction(
        world,
        agent_id=1,
        probability=0.9,
        horizon=10
    )
    print(f"  [OK] Prediction: p=0.9, horizon=10")

    # Advance to horizon WITHOUT completing task
    print("\n=== Tick 1-10: Time passes (no task completion) ===")
    world = run_n_ticks(world, 10)
    assert world.tick == 10
    print(f"  [OK] Reached horizon at tick {world.tick}")

    # Verify prediction resolved as FALSE
    print("\n=== Tick 10: Prediction Resolves (Failed) ===")
    resolution_events = [e for e in world.log.completed if e.event_type == "resolution"]
    assert len(resolution_events) == 1

    resolution = resolution_events[0]
    assert resolution.payload["outcome"] == False
    assert abs(resolution.payload["brier_score"] - 0.81) < 0.001  # (0.9 - 0.0)^2
    print(f"  [OK] Resolution: outcome={resolution.payload['outcome']}")
    print(f"  [OK] Brier score: {resolution.payload['brier_score']:.4f} (overconfident!)")

    # Calibration should reflect the bad prediction
    calibration = world.state.node_attrs["calibration"][1]
    assert abs(calibration - 0.81) < 0.001
    print(f"  [OK] Agent calibration: {calibration:.4f} (needs improvement)")


def test_multiple_predictions_running_average():
    """
    Test Case 3: Multiple predictions -> running average calibration

    Flow:
    - Prediction 1: p=0.9, outcome=False -> Brier=0.81
    - Prediction 2: p=0.7, outcome=True -> Brier=0.09
    - Calibration = (0.81 + 0.09) / 2 = 0.45
    """
    world = create_world(n_agents=2)

    print("\n=== Prediction 1: Overconfident ===")
    world = register_prediction(world, agent_id=1, probability=0.9, horizon=10)
    world = run_n_ticks(world, 10)  # Don't complete task

    calibration_1 = world.state.node_attrs["calibration"][1]
    assert abs(calibration_1 - 0.81) < 0.001
    print(f"  [OK] After prediction 1: calibration={calibration_1:.4f}")

    print("\n=== Prediction 2: Better calibrated ===")
    world = register_prediction(world, agent_id=1, probability=0.7, horizon=20)
    world = mark_task_complete(world, agent_id=1)
    world = run_n_ticks(world, 10)  # Advance to tick 20

    calibration_2 = world.state.node_attrs["calibration"][1]
    expected = (0.81 + 0.09) / 2  # Running average
    assert abs(calibration_2 - expected) < 0.001
    print(f"  [OK] After prediction 2: calibration={calibration_2:.4f}")
    print(f"  [OK] Running average working correctly")

    # Verify prediction count
    assert world.state.node_attrs["prediction_count"][1] == 2


def test_pipeline_composition():
    """
    Test Case 4: Verify the pipeline model works

    Shows that transformations compose cleanly:
    - Each function is World -> World
    - Functions can be chained
    - The tick pipeline is itself a composition
    """
    world = create_world(n_agents=2)

    print("\n=== Testing Pipeline Composition ===")

    # Each of these is World -> World
    world = register_prediction(world, agent_id=1, probability=0.8, horizon=5)
    assert world.tick == 0
    print(f"  [OK] After register_prediction: tick={world.tick}")

    world = mark_task_complete(world, agent_id=1)
    assert world.tick == 0  # These don't advance tick directly
    print(f"  [OK] After mark_task_complete: tick={world.tick}")

    # tick_pipeline advances time AND resolves predictions
    world = tick_pipeline(world)
    assert world.tick == 1
    print(f"  [OK] After tick_pipeline: tick={world.tick}")

    # Can compose manually
    world = tick_pipeline(tick_pipeline(tick_pipeline(world)))
    assert world.tick == 4
    print(f"  [OK] After 3 more ticks: tick={world.tick}")

    # Or use run_n_ticks (which is a fold)
    world = run_n_ticks(world, 1)
    assert world.tick == 5
    print(f"  [OK] After run_n_ticks(1): tick={world.tick}")

    # Prediction should have resolved at tick 5
    resolutions = [e for e in world.log.completed if e.event_type == "resolution"]
    assert len(resolutions) == 1
    print(f"  [OK] Prediction resolved at horizon")

    print("\n  [OK] Pipeline model works: all functions are World -> World")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("END-TO-END TEST 1: THE CORE PREDICTION CYCLE")
    print("="*70)

    test_prediction_cycle_success()
    print("\n" + "="*70)

    test_prediction_cycle_failure()
    print("\n" + "="*70)

    test_multiple_predictions_running_average()
    print("\n" + "="*70)

    test_pipeline_composition()
    print("\n" + "="*70)

    print("\n[PASS] ALL TESTS PASSED")
    print("\nThe core prediction cycle works:")
    print("  • Predictions register")
    print("  • Resolutions happen at horizon")
    print("  • Brier scores computed correctly")
    print("  • Calibration updates as running average")
    print("  • Pipeline model: all functions are World -> World")
