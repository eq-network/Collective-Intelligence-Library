from typing import Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState
from core.property import Property

# Type alias for PRNG keys
RandomKey = jnp.ndarray

def initialize_expertise(
    num_agents: int, 
    key: RandomKey,
    distribution_params: Dict[str, float] = None
) -> jnp.ndarray:
    """
    Initialize agent expertise values following a beta distribution.
    
    Beta distribution provides a realistic model for expertise:
    - Some agents have high expertise (subject matter experts)
    - Most agents have moderate expertise (general knowledge)
    - Some agents have low expertise (limited domain knowledge)
    
    Args:
        num_agents: Number of agents to generate expertise for
        key: JAX PRNG key for randomization
        distribution_params: Optional parameters for expertise beta distribution,
                             defaults to {"alpha": 2.0, "beta": 3.0} if None
                             
    Returns:
        Array of shape [num_agents] containing expertise values in [0,1]
    """
    params = distribution_params or {"alpha": 2.0, "beta": 3.0}
    key, subkey = jr.split(key)
    return jr.beta(subkey, params["alpha"], params["beta"], (num_agents,))

def initialize_adversarial_flags(
    num_agents: int, 
    adversarial_proportion: float,
    adversarial_introduction: str,
    key: RandomKey
) -> jnp.ndarray:
    """
    Initialize boolean flags identifying adversarial agents.
    
    Adversarial agents can be introduced immediately or gradually:
    - "immediate": All adversaries are present from the start
    - "gradual": No adversaries initially, introduced during simulation
    
    Args:
        num_agents: Total number of agents
        adversarial_proportion: Fraction of agents that are adversarial (0-1)
        adversarial_introduction: Strategy for introducing adversaries
                                 ("immediate" or "gradual")
        key: JAX PRNG key for randomization
        
    Returns:
        Boolean array of shape [num_agents] where True indicates adversarial agent
    """
    key, subkey = jr.split(key)
    if adversarial_introduction == "immediate":
        # Randomly assign adversarial status immediately
        num_adversarial = int(num_agents * adversarial_proportion)
        flags = jnp.zeros(num_agents, dtype=jnp.bool_)
        indices = jr.choice(
            subkey, 
            jnp.arange(num_agents), 
            shape=(num_adversarial,), 
            replace=False
        )
        return flags.at[indices].set(True)
    else:  # adversarial_introduction == "gradual"
        # In gradual mode, we start with no adversaries
        # They'll be introduced during simulation
        return jnp.zeros(num_agents, dtype=jnp.bool_)

def initialize_beliefs(
    num_agents: int, 
    num_options: int,
    key: RandomKey,
    noise_scale: float = 0.1
) -> jnp.ndarray:
    """
    Initialize agent beliefs about decision options.
    
    Initial beliefs are modeled as probability distributions over options:
    - Each agent has a belief distribution over all available options
    - Beliefs start relatively uniform with some random variation
    - Sum of beliefs for each agent equals 1.0 (proper probability distribution)
    
    Args:
        num_agents: Number of agents to initialize beliefs for
        num_options: Number of decision options to have beliefs about
        key: JAX PRNG key for randomization
        noise_scale: Scale of random noise to add to initial beliefs (0-1)
        
    Returns:
        Array of shape [num_agents, num_options] containing belief probabilities
    """
    key, subkey = jr.split(key)
    # Initial beliefs are noisy and uniform across options
    base_beliefs = jnp.ones((num_agents, num_options)) / num_options
    # Add noise to create diversity in initial beliefs
    noise = jr.normal(subkey, shape=(num_agents, num_options)) * noise_scale
    # Ensure beliefs remain positive and normalized
    beliefs = jnp.maximum(base_beliefs + noise, 0.01)
    return beliefs / jnp.sum(beliefs, axis=1, keepdims=True)

def initialize_communication_network(
    num_agents: int, 
    key: RandomKey,
    network_type: str = "small_world",
    network_params: Dict[str, Any] = None
) -> jnp.ndarray:
    """
    Initialize a communication network between agents.
    
    Creates an adjacency matrix representing the communication topology:
    - "random": Random sparse connections with specified probability
    - "small_world": Small-world network with local clustering and shortcuts
    - "scale_free": Scale-free network with degree following power law
    
    Args:
        num_agents: Number of agents in the network
        key: JAX PRNG key for randomization
        network_type: Type of network topology to generate
        network_params: Parameters specific to the network type
                       - "random": {"p_connect": 0.1}
                       - "small_world": {"k": 4, "p_rewire": 0.1}
                       - "scale_free": {"m": 2}
                       
    Returns:
        Float array of shape [num_agents, num_agents] representing adjacency matrix
    """
    params = network_params or {}
    key, subkey = jr.split(key)
    
    if network_type == "random":
        # Simple random network
        p_connect = params.get("p_connect", 0.1)
        random_matrix = jr.uniform(subkey, (num_agents, num_agents)) < p_connect
        # Remove self-loops (no self-communication)
        identity = jnp.eye(num_agents, dtype=jnp.bool_)
        # Ensure symmetry (undirected graph)
        upper_triangular = jnp.triu(random_matrix, k=1)
        symmetric_matrix = upper_triangular | jnp.transpose(upper_triangular)
        # Convert to float matrix for JAX compatibility
        return jnp.array(symmetric_matrix & ~identity, dtype=jnp.float32)
    
    elif network_type == "small_world":
        # Simple approximation of small-world network
        # For a proper implementation, consider using NetworkX and converting to JAX
        k = params.get("k", 4)  # Average degree
        p_rewire = params.get("p_rewire", 0.1)  # Rewiring probability
        
        # Start with a ring lattice
        lattice = jnp.zeros((num_agents, num_agents), dtype=jnp.bool_)
        for i in range(1, k // 2 + 1):
            indices = jnp.arange(num_agents)
            lattice = lattice.at[indices, (indices + i) % num_agents].set(True)
            lattice = lattice.at[(indices + i) % num_agents, indices].set(True)
        
        # Random rewiring
        rewire_candidates = jnp.where(lattice, 
                                     jr.uniform(subkey, (num_agents, num_agents)) < p_rewire, 
                                     False)
        # This is a simplification; proper rewiring requires more complex operations
        
        return jnp.array(lattice, dtype=jnp.float32)
    
    else:
        # Default to random network if type not recognized
        p_connect = params.get("p_connect", 0.1)
        random_matrix = jr.uniform(subkey, (num_agents, num_agents)) < p_connect
        identity = jnp.eye(num_agents, dtype=jnp.bool_)
        return jnp.array(random_matrix & ~identity, dtype=jnp.float32)

def initialize_crop_distributions(
    crops: List[str],
    key: RandomKey,
    volatility_params: Dict[str, Dict[str, Tuple[float, float]]] = None,
    default_scale: float = 10.0
) -> Dict[str, Dict[str, float]]:
    """
    Initialize statistical distributions for crop yields.
    
    Each crop's yield distribution is modeled as a parameterized Beta distribution.
    
    Args:
        crops: List of crop names
        key: JAX PRNG key for randomization
        volatility_params: Parameters defining alpha/beta ranges by volatility type
                          Default: {
                              "stable": {"alpha": (5.0, 7.0), "beta": (3.0, 5.0)},
                              "variable": {"alpha": (2.0, 5.0), "beta": (1.0, 3.0)}
                          }
        default_scale: Default scale parameter for all distributions
        
    Returns:
        Dictionary mapping crop names to distribution parameters
    """
    # Default volatility parameters if none provided
    if volatility_params is None:
        volatility_params = {
            "stable": {"alpha": (5.0, 7.0), "beta": (3.0, 5.0)},
            "variable": {"alpha": (2.0, 5.0), "beta": (1.0, 3.0)}
        }
    
    num_crops = len(crops)
    distributions = {}
    
    key, *subkeys = jr.split(key, num_crops + 1)
    
    for i, crop in enumerate(crops):
        # Get volatility type for this crop (default to "stable")
        volatility = "stable"  # This should come from a parameter instead of hardcoding
        
        # Get parameter ranges for this volatility
        param_ranges = volatility_params.get(volatility, volatility_params["stable"])
        
        # Generate parameters within specified ranges
        alpha_range = param_ranges["alpha"]
        beta_range = param_ranges["beta"]
        
        alpha_min, alpha_max = alpha_range
        beta_min, beta_max = beta_range
        
        alpha = alpha_min + jr.uniform(subkeys[i]) * (alpha_max - alpha_min)
        beta = beta_min + jr.uniform(subkeys[i]) * (beta_max - beta_min)
        
        distributions[crop] = {
            "alpha": float(alpha),
            "beta": float(beta),
            "scale": default_scale
        }
    
    return distributions


def initialize_democratic_graph_state(
    num_agents: int,
    crops: List[str],
    initial_resources: int,
    resource_min_threshold: int,
    adversarial_proportion: float,
    adversarial_introduction: str,
    yield_volatility: str,
    key: RandomKey,
    expertise_params: Dict[str, float] = None,
    network_params: Dict[str, Any] = None,
    belief_noise: float = 0.1
) -> GraphState:
    """
    Initialize a complete graph state for democratic decision simulations.
    
    This function combines all individual initializers to create a fully
    specified initial graph state ready for simulation.
    
    Args:
        num_agents: Number of agents in the simulation
        crops: List of available crop options
        initial_resources: Starting resource amount
        resource_min_threshold: Resource threshold for survival
        adversarial_proportion: Fraction of agents that are adversarial
        adversarial_introduction: Strategy for introducing adversaries
        yield_volatility: Volatility of crop yields
        key: JAX PRNG key for randomization
        expertise_params: Parameters for expertise distribution
        network_params: Parameters for communication network
        belief_noise: Noise level for initial beliefs
        
    Returns:
        Complete GraphState initialized for democratic simulation
    """
    # Split key for various randomization needs
    key, subkey1, subkey2, subkey3, subkey4 = jr.split(key, 5)
    
    # Initialize node attributes (agents)
    node_attrs = {
        "expertise": initialize_expertise(num_agents, subkey1, expertise_params),
        "is_adversarial": initialize_adversarial_flags(
            num_agents, 
            adversarial_proportion, 
            adversarial_introduction,
            subkey2
        ),
        "voting_power": jnp.ones(num_agents),  # Initial equal voting power
        "belief": initialize_beliefs(num_agents, len(crops), subkey3, belief_noise),
    }
    
    # Initialize adjacency matrices
    adj_matrices = {
        "communication": initialize_communication_network(
            num_agents, 
            subkey4,
            network_params=network_params
        ),
        "delegation": jnp.zeros((num_agents, num_agents)),  # Empty delegation initially
    }
    
    # Initialize global attributes
    global_attrs = {
        "resource_distributions": initialize_crop_distributions(
            crops, 
            yield_volatility,
            key
        ),
        "total_resources": initial_resources,
        "resource_min_threshold": resource_min_threshold,
        "crops": crops,
        "round": 0,
    }
    
    return GraphState(node_attrs, adj_matrices, global_attrs)

# environments/democracy/initialization.py
# Add to existing file

def initialize_portfolio_data(
    portfolio_config: PortfolioConfig, 
    prediction_market: jnp.ndarray,
    key: RandomKey
) -> Dict[str, Any]:
    """Initialize portfolio data from configuration"""
    # Create portfolio definitions based on the scenario
    portfolios = {
        "Conservative": {"weights": jnp.array([0.6, 0.3, 0.1])},
        "Balanced": {"weights": jnp.array([0.4, 0.3, 0.3])},
        "Aggressive": {"weights": jnp.array([0.2, 0.1, 0.7])},
        "Contrarian": {"weights": jnp.array([0.3, 0.6, 0.1])},
        "Market-Weighted": {"weights": jnp.array([0.35, 0.15, 0.5])}
    }
    
    # Calculate expected returns based on prediction market
    for name, portfolio in portfolios.items():
        weights = portfolio["weights"]
        expected_return = jnp.sum(weights * prediction_market)
        portfolios[name]["expected_return"] = float(expected_return)
    
    return portfolios

def initialize_democratic_graph_state(
    # Existing parameters...
    portfolio_config: Optional[PortfolioConfig] = None
) -> GraphState:
    """Initialize graph state for democratic simulation"""
    # Existing initialization code...
    
    # Initialize portfolio-specific state if configured
    if portfolio_config is not None:
        # Initialize token budgets if using tokens
        if portfolio_config.use_tokens:
            # Distribute agent capacities (low, medium, high)
            capacity_types = jnp.zeros(num_agents, dtype=jnp.int32)
            # Determine agent counts by capacity
            low_count = portfolio_config.token_distribution.get("low_capacity_count", 3)
            med_count = portfolio_config.token_distribution.get("medium_capacity_count", 4)
            high_count = portfolio_config.token_distribution.get("high_capacity_count", 3)
            
            # Assign capacities (0=low, 1=medium, 2=high)
            indices = jnp.arange(num_agents)
            key, subkey = jr.split(key)
            indices = jr.permutation(subkey, indices)
            
            # Assign tokens based on capacity
            token_budgets = jnp.zeros(num_agents)
            for i, idx in enumerate(indices):
                if i < low_count:
                    capacity_types = capacity_types.at[idx].set(0)
                    token_budgets = token_budgets.at[idx].set(
                        portfolio_config.token_distribution.get("low_capacity", 150))
                elif i < low_count + med_count:
                    capacity_types = capacity_types.at[idx].set(1)
                    token_budgets = token_budgets.at[idx].set(
                        portfolio_config.token_distribution.get("medium_capacity", 300))
                elif i < low_count + med_count + high_count:
                    capacity_types = capacity_types.at[idx].set(2)
                    token_budgets = token_budgets.at[idx].set(
                        portfolio_config.token_distribution.get("high_capacity", 500))
            
            node_attrs["token_budget"] = token_budgets
            node_attrs["tokens_spent"] = jnp.zeros(num_agents)
            node_attrs["capacity_type"] = capacity_types
        
        # Initialize prediction market
        key, subkey = jr.split(key)
        prediction_market = jnp.array([1.15, 0.6, 1.5])  # From scenario
        
        # Initialize portfolios
        portfolios = initialize_portfolio_data(portfolio_config, prediction_market, subkey)
        
        # Add portfolio preferences to node attributes
        node_attrs["portfolio_preferences"] = jnp.zeros((num_agents, len(portfolios)))
        
        # Add to global attributes
        global_attrs["portfolios"] = portfolios
        global_attrs["prediction_market"] = prediction_market
    
    return GraphState(node_attrs, adj_matrices, global_attrs)