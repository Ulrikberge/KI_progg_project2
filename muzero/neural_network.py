"""
NeuralNetwork: wrapper around a single Flax MLP.
Handles building, forward pass, and parameter initialisation.
Training is handled at the NeuralNetworkManager level (BPTT over all 3 nets).
"""

from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    """Generic multi-layer perceptron."""
    hidden_sizes: Sequence[int]
    output_size: int
    output_activation: str = "linear"  # "linear" | "tanh" | "softmax"

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        if self.output_activation == "tanh":
            x = jnp.tanh(x)
        elif self.output_activation == "softmax":
            x = nn.softmax(x)
        return x


class NeuralNetwork:
    """
    Wraps a Flax MLP.
    Holds the model definition; parameters are managed externally by NNM.
    """

    def __init__(self, hidden_sizes: Sequence[int], output_size: int,
                 output_activation: str = "linear"):
        self.model = MLP(
            hidden_sizes=tuple(hidden_sizes),
            output_size=output_size,
            output_activation=output_activation,
        )

    def init_params(self, rng: jax.Array, input_shape: tuple) -> dict:
        """Initialise and return parameter pytree."""
        dummy = jnp.zeros((1, *input_shape))
        return self.model.init(rng, dummy)

    def forward(self, params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Run the network in forward mode."""
        return self.model.apply(params, x)
