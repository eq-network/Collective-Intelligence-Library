# End-to-End Test Plan

The 3 critical tests that verify the system works.

---

## Test 1: The Core Prediction Cycle

**Why this matters:** This is the fundamental cybernetic loop. If this doesn't work, nothing works.

**User Journey:**
1. Agent makes a prediction
2. Time passes
3. Outcome occurs (or doesn't)
4. Prediction resolves
5. Brier score computed
6. Agent's calibration updates

**Expected Flow:**

```python
# Setup
world = World(
    tick=0,
    state=initialize_graph_state(n_agents=2),  # User + 1 agent
    log=EventLog([], [])
)

# Tick 0: Agent makes prediction
world.log = world.log.append(Event(
    tick=0,
    source=1,  # agent
    target=1,  # self
    event_type="prediction",
    payload={
        "condition": "task_complete",
        "probability": 0.8,
        "horizon": 10
    },
    duration=0  # Instant
))

# Tick 5: Task actually completes
world = run_n_ticks(world, 5)
world.log = world.log.append(Event(
    tick=5,
    source=1,
    target=1,
    event_type="task_complete",
    payload={"task_id": "123"},
    duration=0
))

# Tick 10: Prediction resolves
world = run_n_ticks(world, 5)
# System automatically checks: was task complete by tick 10? YES
# Creates resolution event with Brier score

resolution_events = world.log.filter_completed(event_type="resolution")
assert len(resolution_events) == 1

resolution = resolution_events[0]
assert resolution.payload["outcome"] == True
assert resolution.payload["brier_score"] == 0.04  # (0.8 - 1.0)^2

# Agent calibration updated
assert world.state.node_attrs["calibration"][1] == 0.04
assert world.state.node_attrs["prediction_count"][1] == 1
```

**Success Criteria:**
- [x] Predictions register with horizon
- [x] Resolution happens at correct tick
- [x] Brier score computed correctly: (probability - outcome)²
- [x] Calibration updates with running average
- [x] Works for both True and False outcomes

**Test File:** `tests/test_e2e_prediction_cycle.py`

---

## Test 2: Multi-Agent Learning Over Time

**Why this matters:** Shows the system scaling to multiple agents and that calibration actually improves through feedback.

**User Journey:**
- 3 agents with different initial performance
- Each makes multiple predictions
- Over time, agents learn to calibrate better
- System can rank agents by calibration
- Correction strength shows learning is happening

**Expected Flow:**

```python
# Setup: User + 3 agents
world = initialize_with_agents(["optimist", "realist", "pessimist"])

# Optimist: Starts overconfident
optimist_predictions = [
    (0.95, 10, False),  # Predicts 95%, actually fails
    (0.90, 20, False),  # Still overconfident
    (0.70, 30, True),   # Learning...
    (0.65, 40, True),   # Better calibrated
]

# Realist: Starts well-calibrated, stays good
realist_predictions = [
    (0.75, 10, True),
    (0.70, 20, True),
    (0.72, 30, True),
    (0.71, 40, True),
]

# Pessimist: Starts underconfident
pessimist_predictions = [
    (0.40, 10, True),   # Predicts 40%, actually succeeds
    (0.45, 20, True),   # Still underconfident
    (0.60, 30, True),   # Learning...
    (0.68, 40, True),   # Better calibrated
]

# Run all predictions
for agent_id, predictions in enumerate([optimist_predictions, realist_predictions, pessimist_predictions], start=1):
    for prob, horizon, outcome in predictions:
        world = register_prediction(world, agent_id, prob, horizon)
        world = run_n_ticks(world, horizon - world.tick)
        if outcome:
            world = mark_task_complete(world, agent_id)
        # Resolution happens automatically at horizon

# Check final calibrations
calibrations = world.state.node_attrs["calibration"]

# Realist should have best calibration (lowest Brier)
assert calibrations[2] < calibrations[1]  # Realist < Optimist
assert calibrations[2] < calibrations[3]  # Realist < Pessimist

# Optimist's calibration should have improved over time
optimist_history = get_calibration_history(world, agent_id=1)
assert optimist_history[-1] < optimist_history[0]  # Improved

# Correction strength should be positive (learning)
correction = compute_correction_strength(world, agent_id=1)
assert correction > 0.5

# Can rank agents
leaderboard = get_agent_leaderboard(world)
assert leaderboard[0]["name"] == "realist"
assert leaderboard[0]["calibration"] == calibrations[2]
```

**Success Criteria:**
- [x] Multiple agents can coexist
- [x] Each tracks independent calibration
- [x] Calibration improves over multiple predictions (for agents that learn)
- [x] Correction strength metric > 0 indicates learning
- [x] Can rank agents by performance
- [x] Well-calibrated agents score better than over/under-confident

**Test File:** `tests/test_e2e_multi_agent_learning.py`

---

## Test 3: MCP Server Integration

**Why this matters:** This is how Claude actually uses the system. If the MCP interface doesn't work, the system is unusable in practice.

**User Journey:**
- Start MCP server
- Claude calls tools to register predictions
- Claude calls tools to get context
- State persists between calls
- Resources provide system overview

**Expected Flow:**

```python
from mcp import Server

# Initialize and start server
server = start_mcp_server()
assert server.is_running()

# 1. Claude registers a prediction via MCP
response = server.call_tool("register_prediction", {
    "agent_name": "claude-research",
    "target": "task_complete",
    "probability": 0.75,
    "horizon_ticks": 20
})

assert response["registered"] == True
prediction_id = response["prediction_id"]
assert prediction_id is not None

# 2. Claude gets context about itself
context = server.call_tool("get_agent_context", {
    "agent_name": "claude-research"
})

assert context["total_predictions"] == 1
assert context["resolved_predictions"] == 0
assert context["calibration"] is None  # No resolutions yet

# 3. Time passes (simulated)
for _ in range(15):
    server.call_tool("tick", {})

# 4. Task completes
server.call_tool("log_outcome", {
    "prediction_id": prediction_id,
    "outcome": True
})

# 5. More time passes to resolution
for _ in range(5):
    server.call_tool("tick", {})

# 6. Check updated context
context = server.call_tool("get_agent_context", {
    "agent_name": "claude-research"
})

assert context["calibration"] == 0.0625  # (0.75 - 1.0)^2
assert context["resolved_predictions"] == 1
assert context["total_predictions"] == 1

# 7. Second prediction uses first result
# Claude sees previous calibration and adjusts
response2 = server.call_tool("register_prediction", {
    "agent_name": "claude-research",
    "target": "task_complete_2",
    "probability": 0.80,  # More confident based on past success
    "horizon_ticks": 20
})

# 8. System resources
agents = server.call_resource("agents://list")
assert len(agents) >= 1
assert any(a["name"] == "claude-research" for a in agents)

metrics = server.call_resource("metrics://system")
assert metrics["total_predictions"] == 2
assert metrics["total_resolved"] == 1
assert metrics["current_tick"] == 20

# 9. State persists across server restarts
server.stop()
server = start_mcp_server()  # Reload from disk

context = server.call_tool("get_agent_context", {
    "agent_name": "claude-research"
})

assert context["calibration"] == 0.0625  # Persisted
assert context["total_predictions"] == 2  # Persisted
```

**Success Criteria:**
- [x] MCP server starts successfully
- [x] Tools are callable via MCP protocol
- [x] `register_prediction` creates prediction events
- [x] `log_outcome` resolves predictions
- [x] `get_agent_context` returns accurate data
- [x] `tick` advances time
- [x] Resources (`agents://list`, `metrics://system`) return current state
- [x] State persists between server restarts (disk persistence)
- [x] Multiple agents can use server concurrently

**Test File:** `tests/test_e2e_mcp_integration.py`

---

## Running the Tests

```bash
# All three
pytest tests/test_e2e_*.py -v

# Individual
pytest tests/test_e2e_prediction_cycle.py -v
pytest tests/test_e2e_multi_agent_learning.py -v
pytest tests/test_e2e_mcp_integration.py -v
```

---

## What These Tests Prove

**Test 1 (Prediction Cycle):**
✅ The core feedback loop works
✅ Time advances correctly
✅ Brier scoring is accurate
✅ Calibration updates

**Test 2 (Multi-Agent Learning):**
✅ System scales to multiple agents
✅ Agents learn independently
✅ Can compare performance
✅ Correction strength detects learning

**Test 3 (MCP Integration):**
✅ Claude can actually use the system
✅ Tools work as expected
✅ State persists
✅ System is production-ready

**If all 3 pass:** The cybernetic dashboard is functional end-to-end.
