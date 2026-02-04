"""
Standard RL components (Policy, Encoder implementations).

Uses existing libraries - no custom RL code.
"""
import jax.numpy as jnp
from jax import random
from typing import Dict, Any


# ============================================================================
# OBSERVATION ENCODERS
# ============================================================================

class FlatEncoder:
    """
    Simple flat encoder: concatenates all observation values.

    observation dict → flatten → array
    """
    def encode(self, observation: Dict[str, Any]) -> jnp.ndarray:
        """Flatten observation dict to array."""
        values = []
        for key in sorted(observation.keys()):  # Sorted for consistency
            val = observation[key]
            if isinstance(val, (int, float)):
                values.append(float(val))
            elif hasattr(val, '__iter__'):  # Array-like
                values.extend(jnp.asarray(val).flatten())
        return jnp.array(values)


# ============================================================================
# POLICIES (Use existing RL libraries here)
# ============================================================================

class RandomPolicy:
    """Random baseline policy."""
    def __init__(self, action_dim: int = 1):
        self.action_dim = action_dim

    def __call__(self, observation: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        """Random action in [0, 1]."""
        return random.uniform(key, shape=(self.action_dim,))


class LinearPolicy:
    """
    Simple linear policy: action = W @ obs

    Interpretable, easy to initialize with strategies.
    """
    def __init__(self, weights: jnp.ndarray):
        self.weights = weights

    def __call__(self, observation: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        """Linear: a = W @ obs"""
        action = jnp.dot(self.weights, observation)
        return jnp.maximum(action, 0.0)  # Non-negative


# ============================================================================
# STRATEGY PRESETS (Named Configurations)
# ============================================================================

def create_strategy(name: str, obs_dim: int) -> LinearPolicy:
    """
    Create named strategy as linear policy.

    Strategies = different weight configurations.
    """
    weights = jnp.zeros(obs_dim)

    if name == "cooperate":
        # Send fraction of my resources
        weights = weights.at[0].set(0.1)

    elif name == "defect":
        # Send nothing
        pass  # Already zeros

    elif name == "tit_for_tat":
        # Mirror what they sent
        weights = weights.at[2].set(1.0)  # Assumes obs[2] = "they sent me"

    return LinearPolicy(weights)
