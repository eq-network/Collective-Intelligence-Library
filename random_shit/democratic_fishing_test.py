"""
Democratic Fishing System - Inspired by Red Teaming Democracy Thesis

Implements:
1. Multiple fish species (resources)
2. Portfolio-based voting on extraction policies
3. Three democratic mechanisms (PDD, PRD, PLD)
4. Messaging between agents
5. Adversarial and aligned agents
6. Prediction markets for information

Environment:
- Shared fishing ground with 3 fish species
- Agents vote on extraction policies (portfolios)
- Each portfolio specifies catch rates for each species
- Fish regenerate based on population levels
- Agents can send messages to influence others
- Collective survival depends on sustainable fishing
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

from core import GraphState, initialize_graph_state, get_observation, Policy

# ============================================================================
# PORTFOLIO DEFINITIONS (Extraction Policies)
# ============================================================================

@dataclass
class Portfolio:
    """Extraction policy for different fish species."""
    name: str
    extraction_rates: jnp.ndarray  # (n_species,) - fraction to extract
    description: str

# Define 4 extraction portfolios
PORTFOLIOS = [
    Portfolio(
        name="Conservative",
        extraction_rates=jnp.array([0.05, 0.05, 0.05]),
        description="Low extraction across all species"
    ),
    Portfolio(
        name="Balanced",
        extraction_rates=jnp.array([0.10, 0.10, 0.10]),
        description="Moderate extraction across all species"
    ),
    Portfolio(
        name="Aggressive",
        extraction_rates=jnp.array([0.20, 0.20, 0.20]),
        description="High extraction across all species"
    ),
    Portfolio(
        name="Species_Focused",
        extraction_rates=jnp.array([0.05, 0.05, 0.25]),
        description="Focus on high-value species (cod)"
    ),
]

# ============================================================================
# DEMOCRATIC MECHANISMS
# ============================================================================

def direct_democracy_vote(votes: jnp.ndarray) -> int:
    """
    Predictive Direct Democracy (PDD): All votes equal weight.

    Args:
        votes: (n_agents, n_portfolios) approval votes

    Returns:
        Winning portfolio index
    """
    total_votes = jnp.sum(votes, axis=0)
    return int(jnp.argmax(total_votes))


def representative_democracy_vote(
    votes: jnp.ndarray,
    representatives: jnp.ndarray
) -> int:
    """
    Predictive Representative Democracy (PRD): Only representatives vote.

    Args:
        votes: (n_agents, n_portfolios) approval votes
        representatives: (n_representatives,) indices of representatives

    Returns:
        Winning portfolio index
    """
    rep_votes = votes[representatives]
    total_votes = jnp.sum(rep_votes, axis=0)
    return int(jnp.argmax(total_votes))


def liquid_democracy_vote(
    votes: jnp.ndarray,
    delegations: jnp.ndarray,
    performance_history: jnp.ndarray
) -> int:
    """
    Predictive Liquid Democracy (PLD): Performance-based delegation.

    Args:
        votes: (n_agents, n_portfolios) approval votes
        delegations: (n_agents,) -1=vote directly, else delegate to agent_id
        performance_history: (n_agents,) cumulative performance scores

    Returns:
        Winning portfolio index
    """
    # Calculate weighted votes based on delegations
    vote_weights = jnp.ones(len(votes))

    # Add delegation weights
    for i in range(len(delegations)):
        if delegations[i] >= 0:  # Is delegating
            delegate_id = int(delegations[i])
            # Transfer vote weight to delegate
            vote_weights = vote_weights.at[delegate_id].add(vote_weights[i])
            vote_weights = vote_weights.at[i].set(0)

    # Weight votes by delegation power
    weighted_votes = votes * vote_weights[:, None]
    total_votes = jnp.sum(weighted_votes, axis=0)
    return int(jnp.argmax(total_votes))


# ============================================================================
# PREDICTION MARKET SIMULATION
# ============================================================================

def generate_market_signals(
    portfolios: List[Portfolio],
    fish_populations: jnp.ndarray,
    key: random.PRNGKey,
    noise_level: float = 0.15
) -> jnp.ndarray:
    """
    Simulate prediction market signals about portfolio outcomes.

    Args:
        portfolios: List of available portfolios
        fish_populations: Current fish populations
        key: PRNGKey for noise
        noise_level: Standard deviation of noise

    Returns:
        Predicted yields for each portfolio (noisy)
    """
    n_portfolios = len(portfolios)

    # Calculate true expected yields based on fish populations
    true_yields = []
    for portfolio in portfolios:
        # Yield depends on extraction rate and fish availability
        # Lower extraction = higher regeneration = better long-term yield
        sustainability_score = 1.0 - jnp.mean(portfolio.extraction_rates)
        population_health = jnp.mean(fish_populations) / 1000.0  # Normalize

        true_yield = 1.0 + (sustainability_score * 0.3 * population_health)
        true_yields.append(true_yield)

    true_yields = jnp.array(true_yields)

    # Add noise to simulate imperfect market
    noise = random.normal(key, (n_portfolios,)) * noise_level
    noisy_signals = true_yields + noise

    return noisy_signals


# ============================================================================
# AGENT POLICIES
# ============================================================================

class AlignedVoter:
    """
    Aligned agent: Tries to maximize collective resources.
    Interprets market signals and votes for best portfolio.
    """
    def __init__(self, agent_id: int, n_portfolios: int):
        self.agent_id = agent_id
        self.n_portfolios = n_portfolios

    def __call__(
        self,
        obs: jnp.ndarray,
        key: random.PRNGKey,
        market_signals: jnp.ndarray
    ) -> Tuple[jnp.ndarray, str]:
        """
        Returns: (votes, message)
        votes: approval votes for portfolios
        message: text message to send to other agents
        """
        # Vote for portfolios with above-average predicted yield
        mean_signal = jnp.mean(market_signals)
        votes = (market_signals > mean_signal).astype(jnp.float32)

        # Ensure at least one vote
        if jnp.sum(votes) == 0:
            best_idx = jnp.argmax(market_signals)
            votes = votes.at[best_idx].set(1.0)

        # Create message encouraging cooperation
        best_portfolio = int(jnp.argmax(market_signals))
        message = f"Agent{self.agent_id}: Portfolio {best_portfolio} looks best for long-term"

        return votes, message


class AdversarialVoter:
    """
    Adversarial agent: Tries to minimize collective resources.
    Votes for worst portfolios and sends misleading messages.
    """
    def __init__(self, agent_id: int, n_portfolios: int):
        self.agent_id = agent_id
        self.n_portfolios = n_portfolios

    def __call__(
        self,
        obs: jnp.ndarray,
        key: random.PRNGKey,
        market_signals: jnp.ndarray
    ) -> Tuple[jnp.ndarray, str]:
        """
        Returns: (votes, message)
        """
        # Vote for portfolios with lowest predicted yield
        mean_signal = jnp.mean(market_signals)
        votes = (market_signals < mean_signal).astype(jnp.float32)

        # Ensure at least one vote
        if jnp.sum(votes) == 0:
            worst_idx = jnp.argmin(market_signals)
            votes = votes.at[worst_idx].set(1.0)

        # Send misleading message
        worst_portfolio = int(jnp.argmin(market_signals))
        message = f"Agent{self.agent_id}: Portfolio {worst_portfolio} will maximize yield!"

        return votes, message


# ============================================================================
# SIMULATION ENVIRONMENT
# ============================================================================

class DemocraticFishingEnvironment:
    """
    Multi-species fishing with democratic policy selection.
    """
    def __init__(
        self,
        n_agents: int = 10,
        n_species: int = 3,
        initial_fish_per_species: float = 1000.0,
        democratic_mechanism: str = "PDD",  # PDD, PRD, PLD
        adversarial_proportion: float = 0.2,
        n_representatives: int = 3,
        survival_threshold: float = 100.0
    ):
        self.n_agents = n_agents
        self.n_species = n_species
        self.initial_fish_per_species = initial_fish_per_species
        self.mechanism = democratic_mechanism
        self.adversarial_proportion = adversarial_proportion
        self.n_representatives = n_representatives
        self.survival_threshold = survival_threshold

        # Species names
        self.species_names = ["Herring", "Salmon", "Cod"]

        # Species regeneration rates (per round)
        self.regeneration_rates = jnp.array([0.08, 0.05, 0.03])  # Fast, medium, slow

        # Species values (resource value per unit caught)
        self.species_values = jnp.array([1.0, 2.0, 5.0])  # Low, medium, high value

        # Initialize agents
        n_adversarial = int(n_agents * adversarial_proportion)
        self.agent_types = ["adversarial"] * n_adversarial + ["aligned"] * (n_agents - n_adversarial)

        # Create agent policies
        self.agents = []
        for i, agent_type in enumerate(self.agent_types):
            if agent_type == "aligned":
                self.agents.append(AlignedVoter(i, len(PORTFOLIOS)))
            else:
                self.agents.append(AdversarialVoter(i, len(PORTFOLIOS)))

        # For PRD: select representatives (first n_representatives agents)
        if mechanism == "PRD":
            self.representatives = jnp.arange(n_representatives)
        else:
            self.representatives = None

        # For PLD: track performance history
        self.performance_history = jnp.zeros(n_agents)

    def reset(self, key: random.PRNGKey) -> Dict:
        """Initialize environment state."""
        # Fish populations for each species
        fish_populations = jnp.ones(self.n_species) * self.initial_fish_per_species

        # Agent resources (collective pool)
        collective_resources = 100.0

        # Message history
        messages = []

        # Voting history
        voting_history = []

        state = {
            "fish_populations": fish_populations,
            "collective_resources": collective_resources,
            "messages": messages,
            "voting_history": voting_history,
            "round": 0,
            "alive": True
        }

        return state

    def step(
        self,
        state: Dict,
        key: random.PRNGKey
    ) -> Tuple[Dict, Dict]:
        """
        One round of democratic fishing.

        Returns:
            new_state, info
        """
        key, *subkeys = random.split(key, 4)

        # 1. Generate market signals
        market_signals = generate_market_signals(
            PORTFOLIOS,
            state["fish_populations"],
            subkeys[0],
            noise_level=0.15
        )

        # 2. Agents observe and vote
        votes = []
        messages = []

        for i, agent in enumerate(self.agents):
            # Simple observation: fish populations + market signals
            obs = jnp.concatenate([
                state["fish_populations"],
                market_signals
            ])

            agent_votes, message = agent(obs, subkeys[1], market_signals)
            votes.append(agent_votes)
            messages.append(message)

        votes = jnp.array(votes)  # (n_agents, n_portfolios)

        # 3. Aggregate votes based on mechanism
        if self.mechanism == "PDD":
            winning_portfolio_idx = direct_democracy_vote(votes)
        elif self.mechanism == "PRD":
            winning_portfolio_idx = representative_democracy_vote(
                votes, self.representatives
            )
        elif self.mechanism == "PLD":
            # For now, no delegations (simplified)
            delegations = jnp.ones(self.n_agents) * -1  # All vote directly
            winning_portfolio_idx = liquid_democracy_vote(
                votes, delegations, self.performance_history
            )
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")

        chosen_portfolio = PORTFOLIOS[winning_portfolio_idx]

        # 4. Apply extraction
        extraction = state["fish_populations"] * chosen_portfolio.extraction_rates
        new_fish = state["fish_populations"] - extraction

        # 5. Calculate resources gained
        resources_gained = jnp.sum(extraction * self.species_values)
        new_collective_resources = state["collective_resources"] + resources_gained

        # 6. Fish regeneration
        # Logistic growth: r * N * (1 - N/K)
        carrying_capacity = self.initial_fish_per_species * 2
        growth = (
            self.regeneration_rates * new_fish *
            (1 - new_fish / carrying_capacity)
        )
        new_fish = new_fish + growth
        new_fish = jnp.maximum(new_fish, 0.0)  # Can't be negative

        # 7. Check survival
        alive = (
            new_collective_resources > self.survival_threshold and
            jnp.all(new_fish > 1.0)  # At least 1 fish per species
        )

        # Create new state
        new_state = {
            "fish_populations": new_fish,
            "collective_resources": float(new_collective_resources),
            "messages": messages,
            "voting_history": state["voting_history"] + [winning_portfolio_idx],
            "round": state["round"] + 1,
            "alive": bool(alive)
        }

        # Info for analysis
        info = {
            "portfolio_chosen": chosen_portfolio.name,
            "extraction": extraction,
            "resources_gained": float(resources_gained),
            "market_signals": market_signals,
            "fish_populations": new_fish
        }

        return new_state, info


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DEMOCRATIC FISHING SYSTEM - THESIS REPLICATION")
    print("=" * 80)

    # Test all three mechanisms
    mechanisms = ["PDD", "PRD", "PLD"]
    adversarial_levels = [0.0, 0.2, 0.5]

    for mechanism in mechanisms:
        print(f"\n{'=' * 80}")
        print(f"Testing: {mechanism}")
        print(f"{'=' * 80}")

        for adv_prop in adversarial_levels:
            print(f"\nAdversarial Proportion: {adv_prop:.0%}")

            key = random.PRNGKey(42)
            env = DemocraticFishingEnvironment(
                n_agents=10,
                democratic_mechanism=mechanism,
                adversarial_proportion=adv_prop
            )

            state = env.reset(key)

            n_rounds = 30
            resources_history = [state["collective_resources"]]

            for round_idx in range(n_rounds):
                key, subkey = random.split(key)
                state, info = env.step(state, subkey)
                resources_history.append(state["collective_resources"])

                if not state["alive"]:
                    print(f"  COLLAPSE at round {round_idx}")
                    break

                if round_idx % 10 == 0:
                    print(f"  Round {round_idx}: Resources={state['collective_resources']:.1f}, "
                          f"Fish={jnp.mean(state['fish_populations']):.1f}")

            final_resources = resources_history[-1]
            initial_resources = resources_history[0]
            growth = ((final_resources / initial_resources) - 1) * 100

            print(f"  Final: {final_resources:.1f} ({growth:+.1f}%)")
            print(f"  Survival: {'YES' if state['alive'] else 'NO'}")
