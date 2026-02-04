"""
Simple game-theoretic policies.
"""
import jax.numpy as jnp
from jax import random


class RandomPolicy:
    """Random action policy."""

    def __init__(self, output_shape: tuple):
        self.output_shape = output_shape

    def __call__(self, obs: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        return random.uniform(key, self.output_shape)


class TitForTatPolicy:
    """
    Tit-for-tat: cooperate if they cooperated, defect if they defected.

    Reads last messages from observation and mirrors them.
    """

    def __init__(self, output_shape: tuple, obs_split_idx: int):
        """
        Args:
            output_shape: Shape of action output
            obs_split_idx: Index where edge data starts in observation
        """
        self.output_shape = output_shape
        self.obs_split_idx = obs_split_idx

    def __call__(self, obs: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        # Extract last messages received from observation
        edge_data = obs[self.obs_split_idx:]

        # Mirror the last message (tit-for-tat)
        if edge_data.size >= jnp.prod(jnp.array(self.output_shape)):
            action = edge_data[:jnp.prod(jnp.array(self.output_shape))].reshape(self.output_shape)
        else:
            action = jnp.zeros(self.output_shape)

        return action
