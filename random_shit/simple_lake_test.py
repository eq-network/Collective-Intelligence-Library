"""
Simplest Lake Model - Central Resource Hub

Architecture:
- Node 0: Lake (central resource hub with fish)
- Nodes 1-N: Agents (fishermen)
- Messages: Extraction requests from agents to lake
- Goal: See collapse with exploiters

Metrics:
- Total fish in system over time
- Average extraction per round
- Survival time
"""
import jax.numpy as jnp
from jax import random
from typing import Dict, List

from core import GraphState, initialize_graph_state

# ============================================================================
# SIMPLE AGENT POLICIES
# ============================================================================

def sustainable_agent(fish_population: float, agent_id: int) -> float:
    """Extract small fixed amount (sustainable)."""
    return 5.0  # Fixed amount per round


def exploiter_agent(fish_population: float, agent_id: int) -> float:
    """Extract 15% of lake fish (unsustainable)."""
    return fish_population * 0.15


# ============================================================================
# LAKE DYNAMICS
# ============================================================================

def regenerate_fish(population: float, regeneration_rate: float = 0.08) -> float:
    """Simple regeneration: 8% growth per round."""
    carrying_capacity = 2000.0
    growth = regeneration_rate * population * (1.0 - population / carrying_capacity)
    return population + growth


# ============================================================================
# SIMULATION
# ============================================================================

def run_lake_simulation(
    n_agents: int,
    n_exploiters: int,
    n_rounds: int,
    initial_lake_fish: float = 1000.0
) -> Dict:
    """
    Run simple lake simulation.

    Args:
        n_agents: Total number of agents
        n_exploiters: Number of exploiter agents
        n_rounds: Number of simulation rounds
        initial_lake_fish: Starting fish in lake

    Returns:
        metrics: Dict with time series data
    """
    # Setup
    lake_fish = initial_lake_fish
    agent_resources = jnp.zeros(n_agents)

    # Agent types: first n_exploiters are exploiters
    agent_policies = []
    for i in range(n_agents):
        if i < n_exploiters:
            agent_policies.append(exploiter_agent)
        else:
            agent_policies.append(sustainable_agent)

    # Metrics
    metrics = {
        "lake_fish": [lake_fish],
        "total_system_fish": [lake_fish],
        "agent_resources": [float(jnp.sum(agent_resources))],
        "round_extractions": [],
        "collapsed": False,
        "collapse_round": None
    }

    # Run simulation
    for round_idx in range(n_rounds):
        # 1. Each agent requests extraction
        extractions = []
        for agent_id, policy in enumerate(agent_policies):
            extraction = policy(lake_fish, agent_id)
            extractions.append(extraction)

        total_extraction = sum(extractions)

        # 2. Lake processes extractions (proportional if over-requested)
        if total_extraction > lake_fish:
            # Scale down proportionally
            scale = lake_fish / total_extraction if total_extraction > 0 else 0
            actual_extractions = [e * scale for e in extractions]
            total_extraction = lake_fish
        else:
            actual_extractions = extractions

        # 3. Update lake
        lake_fish = lake_fish - total_extraction
        lake_fish = max(0.0, lake_fish)

        # 4. Regenerate fish
        lake_fish = regenerate_fish(lake_fish)

        # 5. Update agent resources
        agent_resources = agent_resources + jnp.array(actual_extractions)

        # 6. Record metrics
        total_fish = lake_fish + jnp.sum(agent_resources)
        metrics["lake_fish"].append(lake_fish)
        metrics["total_system_fish"].append(float(total_fish))
        metrics["agent_resources"].append(float(jnp.sum(agent_resources)))
        metrics["round_extractions"].append(total_extraction)

        # 7. Check collapse
        if lake_fish < 10.0:  # Collapse threshold
            metrics["collapsed"] = True
            metrics["collapse_round"] = round_idx
            break

    return metrics


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLE LAKE MODEL - CENTRAL RESOURCE HUB")
    print("=" * 80)

    n_agents = 10
    n_rounds = 100

    # Test different exploiter proportions
    exploiter_proportions = [0.0, 0.2, 0.5, 0.7, 0.9]

    results = []

    for prop in exploiter_proportions:
        n_exploiters = int(n_agents * prop)

        print(f"\n{'=' * 80}")
        print(f"Exploiter Proportion: {prop:.0%} ({n_exploiters}/{n_agents} agents)")
        print(f"{'=' * 80}")

        metrics = run_lake_simulation(
            n_agents=n_agents,
            n_exploiters=n_exploiters,
            n_rounds=n_rounds
        )

        results.append((prop, metrics))

        # Print summary
        print(f"\nInitial lake fish: {metrics['lake_fish'][0]:.1f}")
        print(f"Final lake fish:   {metrics['lake_fish'][-1]:.1f}")
        print(f"Total system fish: {metrics['total_system_fish'][-1]:.1f}")
        print(f"Agent resources:   {metrics['agent_resources'][-1]:.1f}")

        if metrics["collapsed"]:
            print(f"\n[X] COLLAPSE at round {metrics['collapse_round']}")
        else:
            print(f"\n[OK] Survived all {n_rounds} rounds")

        # Print statistics
        avg_extraction = sum(metrics["round_extractions"]) / len(metrics["round_extractions"])
        max_extraction = max(metrics["round_extractions"])
        print(f"\nExtraction stats:")
        print(f"  Average per round: {avg_extraction:.1f}")
        print(f"  Maximum per round: {max_extraction:.1f}")

    # ============================================================================
    # SUMMARY ANALYSIS
    # ============================================================================

    print("\n" + "=" * 80)
    print("SUMMARY: COLLAPSE ANALYSIS")
    print("=" * 80)

    print("\n| Exploiters | Outcome | Collapse Round | Final Lake Fish |")
    print("|------------|---------|----------------|-----------------|")

    for prop, metrics in results:
        outcome = "COLLAPSE" if metrics["collapsed"] else "SURVIVED"
        collapse = metrics["collapse_round"] if metrics["collapsed"] else "-"
        final_fish = metrics["lake_fish"][-1]

        print(f"| {prop:>8.0%} | {outcome:>7} | {str(collapse):>14} | {final_fish:>15.1f} |")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Find collapse threshold
    collapse_threshold = None
    for prop, metrics in results:
        if metrics["collapsed"]:
            collapse_threshold = prop
            break

    if collapse_threshold is not None:
        print(f"\nCollapse threshold: {collapse_threshold:.0%} exploiters")
        print(f"Regeneration rate (8%) cannot sustain extraction above this level")
    else:
        print(f"\nNo collapse observed - system survived even at 90% exploiters")
        print(f"Regeneration rate is sufficient for all tested levels")

    print("\nThis demonstrates:")
    print("1. Central resource hub (lake) model works")
    print("2. Metrics track system health over time")
    print("3. Collapse scenarios emerge at high exploiter concentrations")
    print("4. Tragedy of commons requires threshold exploitation level")
