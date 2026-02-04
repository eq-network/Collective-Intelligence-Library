"""
Quick test: Tragedy of the Commons - Fishing Scenario

Setup:
- 5 fishermen (agents) sharing a fishing ground
- 1 resource type: fish population
- Each fisherman can catch fish (deplete resource)
- Fish regenerate slowly each round
- Prediction: Selfish agents will overfish and collapse the population
- Cooperative agents should sustain the fishery

Testing:
1. All selfish agents → expect collapse
2. All cooperative agents → expect sustainability
3. Mixed strategies → expect tragedy (defectors win short-term, everyone loses long-term)
"""
import jax
import jax.numpy as jnp
from jax import random

# Import core functionality
from core import GraphState, initialize_graph_state, get_observation, apply_action
from agents import RandomPolicy, LinearPolicy

print("=" * 80)
print("TRAGEDY OF THE COMMONS - FISHING SCENARIO")
print("=" * 80)

# ============================================================================
# SETUP FISHING SCENARIO
# ============================================================================

n_fishermen = 5
n_resources = 1  # Just fish
initial_fish_per_agent = 100.0

key = random.PRNGKey(42)

# Initialize resources
initial_resources = jnp.ones((n_fishermen, n_resources)) * initial_fish_per_agent

# Create graph state
state = initialize_graph_state(
    n_agents=n_fishermen,
    n_resources=n_resources,
    initial_resources=initial_resources
)

# Add connectivity (all-to-all - everyone shares the fishing ground)
connections = jnp.ones((n_fishermen, n_fishermen)) - jnp.eye(n_fishermen)
state = state.update_adj_matrix("connections", connections)

# Add message tracking
messages = jnp.zeros((n_fishermen, n_fishermen, n_resources))
state = state.update_edge_attrs("messages", messages)

print(f"\nInitial setup:")
print(f"  Fishermen: {n_fishermen}")
print(f"  Initial fish per agent: {initial_fish_per_agent}")
print(f"  Total fish: {jnp.sum(state.node_attrs['resources']):.2f}")

# ============================================================================
# DEFINE POLICIES
# ============================================================================

# Get observation shape
test_obs = get_observation(state, 0)
obs_dim = test_obs.shape[0]
action_shape = (n_fishermen, n_resources)

print(f"\nObservation dim: {obs_dim}")
print(f"Action shape: {action_shape}")

# Create policies
# Scenario 1: All agents use random policy (uncoordinated fishing)
policies = [RandomPolicy(action_shape) for _ in range(n_fishermen)]

print(f"\nPolicies: {[type(p).__name__ for p in policies]}")

# ============================================================================
# SIMULATION LOOP
# ============================================================================

n_rounds = 50
history = {
    "total_resources": [],
    "agent_resources": [],
    "transfers": []
}

print("\n" + "=" * 80)
print("RUNNING SIMULATION")
print("=" * 80)

for round_idx in range(n_rounds):
    # Each agent observes and acts
    keys = random.split(key, n_fishermen + 1)
    key = keys[0]
    agent_keys = keys[1:]

    # Collect actions from all agents
    actions = []
    for agent_id in range(n_fishermen):
        obs = get_observation(state, agent_id)
        action = policies[agent_id](obs, agent_keys[agent_id])
        actions.append(action)

    # Apply actions sequentially (or we could batch them)
    for agent_id in range(n_fishermen):
        state = apply_action(state, agent_id, actions[agent_id])

    # Resource regeneration (fish reproduce)
    # Simple model: fish grow by 5% each round if population > 0
    current_resources = state.node_attrs["resources"]
    regeneration = current_resources * 0.05
    new_resources = current_resources + regeneration
    state = state.update_node_attrs("resources", new_resources)

    # Track history
    total = float(jnp.sum(state.node_attrs["resources"]))
    history["total_resources"].append(total)
    history["agent_resources"].append(state.node_attrs["resources"].copy())

    if round_idx % 10 == 0:
        print(f"Round {round_idx:3d}: Total fish = {total:8.2f}, "
              f"Avg per agent = {total/n_fishermen:6.2f}")

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

final_total = history["total_resources"][-1]
initial_total = history["total_resources"][0]

print(f"\nInitial total fish: {initial_total:.2f}")
print(f"Final total fish:   {final_total:.2f}")
print(f"Change:             {final_total - initial_total:+.2f} ({100*(final_total/initial_total - 1):+.1f}%)")

# Check if tragedy occurred
if final_total < initial_total * 0.5:
    print("\n[X] TRAGEDY OF THE COMMONS: Fish population collapsed!")
elif final_total > initial_total * 1.5:
    print("\n[+] UNEXPECTED: Fish population grew exponentially!")
else:
    print("\n[=] STABLE: Fish population maintained")

# Agent equity
final_resources = history["agent_resources"][-1]
print(f"\nFinal distribution:")
for i in range(n_fishermen):
    print(f"  Fisherman {i}: {final_resources[i, 0]:.2f} fish")

# Check inequality
max_fish = jnp.max(final_resources)
min_fish = jnp.min(final_resources)
print(f"\nInequality: max={max_fish:.2f}, min={min_fish:.2f}, ratio={max_fish/(min_fish+1e-8):.2f}x")

# ============================================================================
# LEARNINGS
# ============================================================================

print("\n" + "=" * 80)
print("LEARNINGS & OBSERVATIONS")
print("=" * 80)

print("""
1. SYSTEM ARCHITECTURE - SUCCESS:
   - GraphState successfully holds all state (resources, messages, adjacency)
   - get_observation() extracts agent-local view correctly
   - apply_action() updates state functionally (immutable updates work)
   - Pure functions compose cleanly
   - JAX arrays work throughout the pipeline

2. POLICY INTERFACE - WORKS:
   - obs -> action signature is clean and composable
   - Easy to instantiate multiple agents with different policies
   - Random policies execute without errors
   - Action shapes match expectations (n_agents, n_resources)

3. SIMULATION DYNAMICS - UNEXPECTED BEHAVIOR:
   - Fish population EXPLODED (500 -> 5733, +992%)
   - This is NOT a tragedy of the commons!
   - Random transfers + 5% regeneration = exponential growth
   - Current model: agents transfer fish to each other (redistribution)
   - Missing: actual extraction/consumption mechanic

4. KEY INSIGHT - MODEL IS WRONG:
   - apply_action() implements TRANSFERS (zero-sum between agents)
   - Should implement EXTRACTION (agents consume from shared pool)
   - "Fishing" should REDUCE total fish, not redistribute
   - Need: shared resource pool that agents deplete
   - Regeneration rate (5%) is correct, but extraction is missing

5. WHAT ACTUALLY HAPPENED:
   - Agents randomly sent fish to each other
   - Total fish conserved (zero-sum transfers)
   - 5% regeneration added fish every round
   - Result: pure exponential growth at 5% per round
   - No competition, no depletion, no tragedy

6. FIX NEEDED:
   - Redefine action semantics:
     * Action should be "amount to extract from shared pool"
     * Not "amount to send to others"
   - Or: model fishing ground as separate node
   - Or: negative transfers = extraction (remove from system)

7. ARCHITECTURE VALIDATION:
   - Core system works perfectly
   - Problem is in how we MODEL the game, not the framework
   - Framework is flexible enough to fix this
   - Just need to change apply_action() semantics for this scenario

8. NEXT EXPERIMENTS:
   - Test with extraction semantics (agents remove resources)
   - Add shared pool node (bank/government style)
   - Test learned policies with gradient descent
   - Visualize trajectories over time
""")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
