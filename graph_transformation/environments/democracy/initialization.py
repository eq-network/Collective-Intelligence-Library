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