"""
Learnable policies (RL).
"""
import jax.numpy as jnp
from jax import random
from typing import Dict


class LinearPolicy:
    """
    Linear policy: action = W @ obs + b

    Parameters can be learned via gradient descent.
    """

    def __init__(self, input_dim: int, output_shape: tuple):
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.output_dim = jnp.prod(jnp.array(output_shape))

    def __call__(
        self,
        obs: jnp.ndarray,
        key: random.PRNGKey,
        params: Dict[str, jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Args:
            obs: Observation vector
            key: PRNGKey for initialization if needed
            params: {'W': weights, 'b': bias}
        """
        if params is None:
            key_w, key_b = random.split(key)
            params = {
                'W': random.normal(key_w, (self.output_dim, self.input_dim)) * 0.01,
                'b': jnp.zeros(self.output_dim)
            }

        action_flat = params['W'] @ obs + params['b']
        action_flat = jnp.maximum(action_flat, 0.0)

        return action_flat.reshape(self.output_shape)
