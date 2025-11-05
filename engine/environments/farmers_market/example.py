"""
Farmer's Market: Clean example.

State → Agents → Transform → New State
"""
import jax.numpy as jnp
from engine.environments.farmers_market.state import create_simple_farmers_market
from engine.environments.farmers_market.agent_transforms import create_agent_driven_round_transform
from engine.environments.farmers_market.agent_configs import (
    create_diversity_farmer,
    create_accumulator_farmer,
    create_trader_farmer,
    create_random_farmer
)


def run_simulation(num_farmers: int = 10, num_rounds: int = 50, seed: int = 42):
    """Run farmers market simulation."""

    # Initial state
    state = create_simple_farmers_market(num_farmers, seed)

    # Agents (heterogeneous population)
    agents = [
        create_diversity_farmer(),
        create_diversity_farmer(),
        create_accumulator_farmer(),
        create_accumulator_farmer(),
        create_trader_farmer(),
        create_trader_farmer(),
        create_random_farmer(1),
        create_random_farmer(2),
        create_random_farmer(3),
        create_random_farmer(4),
    ]

    # Transform
    step = create_agent_driven_round_transform(agents)

    # Run
    print(f"Running {num_rounds} rounds with {num_farmers} farmers...")
    for _ in range(num_rounds):
        state = step(state)

    # Results
    print(f"\nFinal round: {state.global_attrs['round']}")
    print(f"Total trades: {state.global_attrs.get('total_trades', 0)}")

    for rt in state.global_attrs["resource_types"]:
        total = jnp.sum(state.node_attrs[f"resources_{rt}"])
        print(f"Total {rt}: {total:.1f}")

    return state


if __name__ == "__main__":
    run_simulation()
