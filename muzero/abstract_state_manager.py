"""
AbstractStateManager (ASM): thin interface over the NeuralNetworkManager.
Translates between the RL / MCTS world (which works with abstract states σ)
and the NNM (which holds the actual JAX parameters).

This class is intentionally small — all the heavy lifting is in NNM.
"""

import jax.numpy as jnp
from muzero.neural_network_manager import NeuralNetworkManager


class AbstractStateManager:

    def __init__(self, nnm: NeuralNetworkManager, action_count: int):
        self.nnm = nnm
        self.action_count = action_count

    def get_abstract_state(self, game_state_seq: list) -> jnp.ndarray:
        """
        Map a sequence of q real game states φ = [s_{k-q}, ..., s_k]
        to a single abstract state σ via NNr.
        """
        return self.nnm.represent(game_state_seq)

    def get_policy_and_value(self, abstract_state: jnp.ndarray):
        """
        Given σ, return (policy_logits, value) via NNp.
        The caller should apply softmax to logits if a probability dist is needed.
        """
        return self.nnm.predict(abstract_state)

    def get_next_state_and_reward(self, abstract_state: jnp.ndarray, action: int):
        """
        Given (σ, a), return (σ', predicted_reward) via NNd.
        Used by u-MCTS during node expansion and rollout.
        """
        return self.nnm.dynamics(abstract_state, action)

    def get_legal_actions(self, abstract_state: jnp.ndarray) -> list:
        """All actions are legal in abstract-state space."""
        return list(range(self.action_count))
