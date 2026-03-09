"""
u-MCTS: MuZero Monte Carlo Tree Search.

Operates entirely in abstract-state space — no game-specific code.
Implements the pseudocode from the assignment spec:
  - Tree-policy search (UCB selection)
  - Leaf expansion via NNd
  - DO_ROLLOUT
  - DO_BACKPROPAGATION

Returns π (visit-count distribution) and v* (root value estimate) after
Ms simulations.
"""

import math
import random
import numpy as np
import jax
import jax.numpy as jnp

import config
from muzero.abstract_state_manager import AbstractStateManager


class MCTSNode:
    def __init__(self, abstract_state: jnp.ndarray, parent=None, action_from_parent=None,
                 prior_prob: float = 0.0):
        self.abstract_state = abstract_state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior_prob = prior_prob      # P(a) from NNp at parent

        self.visit_count = 0
        self.total_value = 0.0            # sum of backup values through this node
        self.children: dict[int, "MCTSNode"] = {}
        self.reward = 0.0                 # predicted reward on the edge to this node

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, parent_visits: int, c: float = config.UCB_C) -> float:
        exploration = c * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def update_q_value(self, value: float):
        self.visit_count += 1
        self.total_value += value


class UMCTS:
    """
    u-MCTS as described in the MuZero spec.
    Receives an AbstractStateManager to call NNr/NNd/NNp — no game imports.
    """

    def __init__(self, asm: AbstractStateManager):
        self.asm = asm

    def search(self, root_abstract_state: jnp.ndarray, n_simulations: int,
               dmax: int) -> tuple[np.ndarray, float]:
        """
        Run Ms simulations from the given root abstract state.
        Returns:
          π  — normalised visit-count distribution over actions
          v* — value estimate of root node
        """
        root = MCTSNode(root_abstract_state)

        # Initialise root prior probs via NNp
        self._expand(root)

        for _ in range(n_simulations):
            leaf, depth = self._tree_policy(root)

            # DO_ROLLOUT from a random child of the leaf
            if leaf.is_leaf():
                self._expand(leaf)
            if leaf.children:
                child_star = random.choice(list(leaf.children.values()))
            else:
                child_star = leaf

            accum_rewards = self._do_rollout(child_star, dmax - depth)
            self._do_backpropagation(child_star, root, accum_rewards)

        # π = normalised visit counts over root's children
        actions = list(root.children.keys())
        n_actions = self.asm.action_count
        visits = np.zeros(n_actions)
        for a, child in root.children.items():
            visits[a] = child.visit_count
        total = visits.sum()
        pi = visits / total if total > 0 else np.ones(n_actions) / n_actions

        v_star = root.q_value
        return pi, v_star

    # ------------------------------------------------------------------
    # Internal MCTS steps
    # ------------------------------------------------------------------

    def _tree_policy(self, node: MCTSNode) -> tuple["MCTSNode", int]:
        """Traverse tree using UCB until a leaf is reached. Return (leaf, depth)."""
        depth = 0
        while not node.is_leaf():
            best_action = max(
                node.children,
                key=lambda a: node.children[a].ucb_score(node.visit_count)
            )
            node = node.children[best_action]
            depth += 1
        return node, depth

    def _expand(self, node: MCTSNode):
        """
        Expand node: for each action a, create a child node using NNd.
        Prior probabilities come from NNp applied to node's abstract state.
        """
        policy_logits, _ = self.asm.get_policy_and_value(node.abstract_state)
        priors = jax.nn.softmax(policy_logits)

        for action in self.asm.get_legal_actions(node.abstract_state):
            next_sigma, r_pred = self.asm.get_next_state_and_reward(node.abstract_state, action)
            child = MCTSNode(
                abstract_state=next_sigma,
                parent=node,
                action_from_parent=action,
                prior_prob=float(priors[action]),
            )
            child.reward = float(r_pred)
            node.children[action] = child

    def _do_rollout(self, node: MCTSNode, depth: int) -> list:
        """
        DO_ROLLOUT: simulate forward from node for `depth` steps using NNp/NNd.
        Returns accumulated rewards list (reward per step + final value estimate).
        """
        sigma = node.abstract_state
        accum_rewards = [node.reward]

        for _ in range(max(0, depth)):
            policy_logits, v = self.asm.get_policy_and_value(sigma)
            probs = np.array(jax.nn.softmax(policy_logits))
            action = np.random.choice(len(probs), p=probs)
            sigma, r = self.asm.get_next_state_and_reward(sigma, int(action))
            accum_rewards.append(float(r))

        # Append terminal value estimate
        _, v_leaf = self.asm.get_policy_and_value(sigma)
        accum_rewards.append(float(v_leaf))
        return accum_rewards

    def _do_backpropagation(self, node: MCTSNode, root: MCTSNode, rewards: list):
        """
        DO_BACKPROPAGATION: propagate summed (discounted) rewards from node to root.
        Appends edge rewards while climbing.
        """
        gamma = config.DISCOUNT_GAMMA
        # Compute discounted return from rewards list
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G

        current = node
        while current is not None:
            current.update_q_value(G)
            if current.parent is not None:
                rewards = [current.reward] + rewards
                G = current.reward + gamma * G
            current = current.parent
            if current is root:
                current.update_q_value(G)
                break
