"""
GraphState factory for the fishing commons experiment.

Maps the fishing CPR game onto Mycorrhiza's GraphState abstraction.
100 agents fish from a shared stock with logistic regrowth.
"""
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from core.graph import GraphState


DEFAULT_HARVEST_LEVELS = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def create_initial_state(
    n_agents: int = 100,
    adversarial_fraction: float = 0.0,
    rng_key=None,
) -> GraphState:
    """Create initial GraphState for the fishing commons experiment.

    Agent layout: agents [0, n_adversarial) are adversarial (type 1),
    agents [n_adversarial, n_agents) are cooperative (type 0).

    Parameters
    ----------
    n_agents : int
        Number of agents.
    adversarial_fraction : float
        Fraction of agents that are adversarial (0.0 to 1.0).
    rng_key : optional
        JAX PRNGKey or Python int seed. Defaults to PRNGKey(42).

    Returns
    -------
    GraphState
        Immutable initial state for the fishing commons.
    """
    n_adversarial = int(n_agents * adversarial_fraction)

    # Handle RNG key
    if rng_key is None:
        key = jr.PRNGKey(42)
    elif isinstance(rng_key, (int, np.integer)):
        key = jr.PRNGKey(rng_key)
    else:
        key = rng_key

    key, net_key = jr.split(key)

    # Node types: 0=cooperative, 1=adversarial
    node_types = jnp.zeros(n_agents, dtype=jnp.int32)
    node_types = node_types.at[:n_adversarial].set(1)

    n_harvest_levels = len(DEFAULT_HARVEST_LEVELS)

    node_attrs = {
        "budget": jnp.full(n_agents, 100.0),
        "harvest_weights": jnp.ones((n_agents, n_harvest_levels)),
        "vote_weights": jnp.ones((n_agents, n_harvest_levels)),
        "market_weights": jnp.ones(n_agents),
        "signal": jnp.zeros(n_agents),
        "received_signals": jnp.zeros(n_agents),
        "allocations": jnp.zeros(n_agents),
        "last_harvest": jnp.zeros(n_agents),
        "reward": jnp.zeros(n_agents),
    }

    # Erdos-Renyi random graph, p=0.1
    network = (jr.uniform(net_key, (n_agents, n_agents)) < 0.1).astype(jnp.float32)
    network = network.at[jnp.diag_indices(n_agents)].set(0.0)  # No self-loops
    network = jnp.maximum(network, network.T)  # Symmetrise

    adj_matrices = {"network": network}

    global_attrs = {
        # Dynamic (JAX arrays — traced by JIT)
        "resource_level": jnp.array(4000.0),
        "harvest_target": jnp.array(2.0),
        "penalty_lambda": jnp.array(0.1),
        "clearing_price": jnp.array(0.0),
        "rng_key": key,
        "step": jnp.array(0, dtype=jnp.int32),
        # Static (Python scalars/tuples — baked into compiled kernel)
        "K": 5000.0,
        "r": 0.4,
        "learning_rate": 0.1,
        "n_harvest_levels": n_harvest_levels,  # int, static
    }

    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices=adj_matrices,
        global_attrs=global_attrs,
    )
