# run_portfolio_simulation.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from environments.democracy.configuration import (
    PortfolioDemocracyConfig, CropConfig, PortfolioConfig, TokenBudgetConfig
)
from environments.democracy.portfolio_environment import PortfolioDemocracyEnvironment
from execution.call import create_environment

def run_portfolio_example():
    """Run a simulation based on the portfolio democracy example"""
    # Configure crops based on the Portfolio Example
    crops = [
        CropConfig(
            name="Wheat",
            expected_returns=[0.7, 1.3, 0.6, 1.2],  # Historical returns + current round
            alpha=20.0,
            beta=3.0,
            variance=0.15
        ),
        CropConfig(
            name="Corn",
            expected_returns=[1.5, 0.5, 1.4, 0.5],  # Historical returns + current round
            alpha=7.0,
            beta=5.0,
            variance=0.35
        ),
        CropConfig(
            name="Fungus",
            expected_returns=[0.4, 1.7, 0.3, 1.7],  # Historical returns + current round
            alpha=3.0,
            beta=3.0,
            variance=0.6
        )
    ]
    
    # Create configuration
    config = PortfolioDemocracyConfig(
        mechanism="PLD",  # Test with Liquid Democracy
        num_agents=10,
        num_rounds=15,
        seed=42,
        resources={"initial_amount": 115.0, "threshold": 20.0},
        adversarial={"proportion": 0.2, "introduction": "immediate"},
        crops=crops,
        portfolios=PortfolioConfig(),
        token_budget=TokenBudgetConfig()
    )
    
    # Create environment and run simulation
    env = PortfolioDemocracyEnvironment(config)
    final_state, state_history = env.run_simulation()
    
    # Display results
    summary = env.get_results_summary()
    print(f"Simulation Summary:")
    print(f"Mechanism: {summary['mechanism']}")
    print(f"Rounds Completed: {summary['num_rounds']}")
    print(f"Initial Resources: {summary['initial_resources']:.2f}")
    print(f"Final Resources: {summary['final_resources']:.2f}")
    print(f"Growth Factor: {summary['resource_growth']:.2f}x")
    print("\nDecisions by round:")
    for i, decision in enumerate(summary['decisions']):
        print(f"Round {i+1}: {decision}")
    
    # Plot resource trajectory
    fig = env.plot_resource_trajectory()
    plt.show()
    
    # Run other mechanisms for comparison
    mechanisms = ["PDD", "PRD", "PLD"]
    results = {}
    
    for mechanism in mechanisms:
        mech_config = PortfolioDemocracyConfig(
            mechanism=mechanism,
            num_agents=10,
            num_rounds=15,
            seed=42,
            resources={"initial_amount": 115.0, "threshold": 20.0},
            adversarial={"proportion": 0.2, "introduction": "immediate"},
            crops=crops,
            portfolios=PortfolioConfig(),
            token_budget=TokenBudgetConfig()
        )
        
        mech_env = PortfolioDemocracyEnvironment(mech_config)
        mech_env.run_simulation()
        results[mechanism] = mech_env.get_results_summary()
    
    # Compare final results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mechanism, result in results.items():
        ax.plot(range(len(result['resources'])), result['resources'], marker='o', label=mechanism)
    
    ax.axhline(y=20, color='r', linestyle='--', label='Survival Threshold')
    ax.set_title('Resource Comparison Across Mechanisms')
    ax.set_xlabel('Round')
    ax.set_ylabel('Resources')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    return results

if __name__ == "__main__":
    results = run_portfolio_example()