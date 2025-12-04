"""
Clean Architecture Demo: Shows how core abstractions work together.

This example demonstrates:
1. Environment as scenario factory (creates initial GraphState)
2. Agent as minimal protocol (Observation → Action)
3. Transform composition (pure functions)
4. Property verification (ConservesSum with binding)

Key principles:
- No mutable state
- Pure functions throughout
- Simple, composable abstractions
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
from core.graph import GraphState
from core.agents import Agent, Observation, Action
from core.category import Transform, sequential
from core.property import ConservesSum


# ============================================================================
# 1. ENVIRONMENT: Scenario Factory
# ============================================================================

class SimpleResourceGame:
    """
    Environment as scenario factory.

    Just creates initial GraphState - no runtime state management.
    """

    def create_initial_state(
        self,
        num_agents: int = 5,
        initial_resources: float = 100.0,
        seed: int = 42
    ) -> GraphState:
        """Create initial state for simple resource sharing game."""
        return GraphState(
            node_types=jnp.zeros(num_agents, dtype=jnp.int32),
            node_attrs={
                "resources": jnp.ones(num_agents) * initial_resources,
                "reputation": jnp.ones(num_agents) * 0.5,
            },
            adj_matrices={
                "connections": jnp.ones((num_agents, num_agents)) - jnp.eye(num_agents),
            },
            global_attrs={
                "round": 0,
                "total_trades": 0,
            }
        )


# ============================================================================
# 2. AGENTS: Observation → Action
# ============================================================================

class GenerousAgent:
    """Agent that shares with neighbors who have less."""

    def act(self, observation: Observation) -> Action:
        my_resources = observation["resources"]
        neighbor_avg = observation.get("neighbor_avg_resources", my_resources)

        if my_resources > neighbor_avg * 1.2:
            # Share 10% if I have significantly more
            return {"share_amount": my_resources * 0.1}
        return {"share_amount": 0.0}


class GreedyAgent:
    """Agent that never shares."""

    def act(self, observation: Observation) -> Action:
        return {"share_amount": 0.0}


def random_agent(seed: int) -> Agent:
    """Function-based agent (also valid!)."""
    import random
    rng = random.Random(seed)

    def act(observation: Observation) -> Action:
        my_resources = observation["resources"]
        return {"share_amount": my_resources * rng.uniform(0, 0.15)}

    # Make it match Agent protocol by adding act method
    class FunctionAgent:
        def act(self, obs): return act(obs)

    return FunctionAgent()


# ============================================================================
# 3. TRANSFORMS: Pure functions GraphState → GraphState
# ============================================================================

def create_observation_transform(agents: list[Agent]) -> Transform:
    """Create observations for each agent."""

    def transform(state: GraphState) -> GraphState:
        # Just return state - observations are extracted per-agent as needed
        return state

    return transform


def create_action_transform(agents: list[Agent]) -> Transform:
    """Apply agent actions to update state."""

    def transform(state: GraphState) -> GraphState:
        num_agents = state.num_nodes
        resources = state.node_attrs["resources"]
        connections = state.adj_matrices["connections"]

        # Get actions from all agents
        actions = []
        for i in range(num_agents):
            # Create observation for agent i
            neighbors = jnp.where(connections[i] > 0)[0]
            neighbor_resources = resources[neighbors] if len(neighbors) > 0 else jnp.array([resources[i]])

            obs = {
                "resources": float(resources[i]),
                "neighbor_avg_resources": float(jnp.mean(neighbor_resources)),
            }

            action = agents[i].act(obs)
            actions.append(action)

        # Apply actions (share resources)
        new_resources = resources.copy()
        for i, action in enumerate(actions):
            share_amount = action.get("share_amount", 0.0)
            if share_amount > 0:
                # Deduct from agent
                new_resources = new_resources.at[i].add(-share_amount)

                # Distribute to neighbors
                neighbors = jnp.where(connections[i] > 0)[0]
                if len(neighbors) > 0:
                    per_neighbor = share_amount / len(neighbors)
                    for j in neighbors:
                        new_resources = new_resources.at[j].add(per_neighbor)

        # Update round counter
        new_round = state.global_attrs["round"] + 1

        return state.update_node_attrs("resources", new_resources) \
                    .update_global_attr("round", new_round)

    return transform


def create_round_transform(agents: list[Agent]) -> Transform:
    """Compose observation and action transforms into one round."""
    return sequential(
        create_observation_transform(agents),
        create_action_transform(agents),
    )


# ============================================================================
# 4. PROPERTY VERIFICATION
# ============================================================================

def verify_conservation(initial: GraphState, final: GraphState, tolerance: float = 1e-5):
    """Verify that resources are conserved through simulation."""

    # Create and bind property
    conservation = ConservesSum("resources", tolerance=tolerance)
    bound_conservation = conservation.bind(initial)

    # Check conservation
    is_conserved = bound_conservation.check(final)

    initial_total = float(jnp.sum(initial.node_attrs["resources"]))
    final_total = float(jnp.sum(final.node_attrs["resources"]))

    print(f"\nConservation check:")
    print(f"  Initial total: {initial_total:.2f}")
    print(f"  Final total:   {final_total:.2f}")
    print(f"  Difference:    {abs(final_total - initial_total):.6f}")
    print(f"  Conserved:     {is_conserved}")

    return is_conserved


# ============================================================================
# 5. MAIN SIMULATION
# ============================================================================

def main():
    """Run clean architecture demo."""

    print("=" * 60)
    print("CLEAN ARCHITECTURE DEMO")
    print("=" * 60)

    # 1. Create environment (scenario factory)
    env = SimpleResourceGame()
    initial_state = env.create_initial_state(num_agents=5, initial_resources=100.0)

    print(f"\nInitial state:")
    print(f"  Agents: {initial_state.num_nodes}")
    print(f"  Total resources: {jnp.sum(initial_state.node_attrs['resources']):.2f}")

    # 2. Create heterogeneous agent population
    agents = [
        GenerousAgent(),
        GenerousAgent(),
        GreedyAgent(),
        random_agent(42),
        random_agent(43),
    ]

    print(f"\nAgent types: {[type(a).__name__ for a in agents]}")

    # 3. Create transform
    step = create_round_transform(agents)

    # 4. Run simulation (pure functional loop)
    print(f"\nRunning 10 rounds...")
    state = initial_state
    for round_num in range(10):
        state = step(state)
        if round_num % 3 == 0:
            print(f"  Round {round_num + 1}: resources = {state.node_attrs['resources']}")

    print(f"\nFinal state:")
    print(f"  Round: {state.global_attrs['round']}")
    print(f"  Resources: {state.node_attrs['resources']}")

    # 5. Verify properties
    is_conserved = verify_conservation(initial_state, state)

    if is_conserved:
        print("\n[OK] Resources conserved - architecture working correctly!")
    else:
        print("\n[FAIL] Resources NOT conserved - check transform logic")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
