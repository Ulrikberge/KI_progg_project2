"""
ReinforcementLearningManager (RLM): orchestrates the EPISODE_LOOP.

Implements the pseudocode from the spec:
  1. EH ← ∅
  2. Randomly initialise Ψ   (done in NNM constructor)
  3. For episode in range(Ne):
       a. Reset game → s0
       b. epidata ← ∅
       c. For k in range(Nes):
            - Build φ_k  (lookback window of real states)
            - σ_k = NNr(φ_k)
            - Run Ms u-MCTS simulations → π_k, v*_k
            - Sample a_{k+1} from π_k
            - s_{k+1}, r*_{k+1} = Simulate_game_one_timestep(s_k, a_{k+1})
            - epidata.append([s_k, v*_k, π_k, a_{k+1}, r*_{k+1}])
       d. EH.append(epidata)
       e. if episode % It == 0: DO_BPTT_TRAINING(Ψ, EH, mbs)
  4. Return Ψ

The RLM works almost exclusively with real game states, except when it
calls ASM to get the root abstract state for u-MCTS.
"""

import numpy as np
import jax

import config
from muzero.abstract_state_manager import AbstractStateManager
from muzero.neural_network_manager import NeuralNetworkManager
from muzero.mcts import UMCTS
from muzero.episode_buffer import EpisodeBuffer


class RLManager:

    def __init__(self, game_state_manager, nnm: NeuralNetworkManager,
                 visualizer=None):
        """
        Parameters
        ----------
        game_state_manager : GameStateManager (from games/) passed in — never imported here
        nnm                : NeuralNetworkManager
        visualizer         : optional Visualizer instance
        """
        self.gsm = game_state_manager
        self.nnm = nnm
        self.visualizer = visualizer
        self.asm = AbstractStateManager(nnm, action_count=game_state_manager.get_action_count())
        self.mcts = UMCTS(self.asm)
        self.episode_buffer = EpisodeBuffer()

        self.episode_rewards: list[float] = []
        self.training_losses: list[float] = []

    def run(self):
        """Run the full EPISODE_LOOP. Returns the trained NNM (Ψ)."""
        for episode in range(config.N_EPISODES):
            epidata, total_reward = self._run_episode()
            self.episode_buffer.add_episode(epidata)
            self.episode_rewards.append(total_reward)

            if (episode + 1) % config.TRAINING_INTERVAL == 0:
                loss = self.nnm.train(self.episode_buffer, config.MINIBATCH_SIZE)
                self.training_losses.append(loss)
                print(f"Episode {episode+1:4d}/{config.N_EPISODES}  "
                      f"reward={total_reward:.1f}  loss={loss:.4f}")
            else:
                print(f"Episode {episode+1:4d}/{config.N_EPISODES}  reward={total_reward:.1f}")

            if self.visualizer and config.PLOT_METRICS:
                self.visualizer.plot_training_metrics(
                    episode + 1, self.episode_rewards, self.training_losses
                )

        return self.nnm

    def _run_episode(self) -> tuple[list, float]:
        """
        Run a single episode. Returns (epidata, total_reward).
        epidata: list of (s_k, v*_k, π_k, a_{k+1}, r*_{k+1})
        """
        epidata = []
        state = self.gsm.reset()
        state_history = [state]   # for building φ_k
        total_reward = 0.0

        for k in range(config.STEPS_PER_EPISODE):
            if self.gsm.is_terminal(state):
                break

            # Build φ_k: lookback window of q real game states
            phi_k = self._build_lookback(state_history, config.LOOKBACK)

            # σ_k = NNr(φ_k)
            sigma_k = self.asm.get_abstract_state(phi_k)

            # Run u-MCTS → π_k, v*_k
            pi_k, v_star_k = self.mcts.search(
                sigma_k,
                config.N_MCTS_SIMULATIONS,
                config.MAX_ROLLOUT_DEPTH,
            )

            # Sample a_{k+1} from π_k
            action = int(np.random.choice(len(pi_k), p=pi_k))

            # Simulate one real timestep
            next_state, reward, done = self.gsm.step(state, action)
            total_reward += reward

            # Record episode data
            epidata.append((
                np.array(state, dtype=np.float32),  # s_k
                v_star_k,                            # v*_k
                pi_k,                                # π_k
                action,                              # a_{k+1}
                reward,                              # r*_{k+1}
            ))

            if config.VISUALIZE and self.visualizer:
                self.visualizer.render_game_state(next_state)

            state = next_state
            state_history.append(state)

            if done:
                break

        # Replace MCTS Q-values (v*_k) with actual Monte Carlo returns.
        # MCTS values are noise when the networks are untrained. Using the real
        # discounted return gives meaningful value targets from episode 1.
        # The policy targets (π_k) still come from MCTS — only v*_k changes.
        G = 0.0
        for k in range(len(epidata) - 1, -1, -1):
            s_k, _, pi_k, a_k, r_k = epidata[k]
            G = r_k + config.DISCOUNT_GAMMA * G
            epidata[k] = (s_k, G, pi_k, a_k, r_k)

        return epidata, total_reward

    @staticmethod
    def _build_lookback(state_history: list, q: int) -> list:
        """Return the last q states, padding with zeros at the start if needed."""
        if len(state_history) == 0:
            raise ValueError("state_history is empty")
        state_dim = np.array(state_history[0]).shape
        result = []
        for i in range(q):
            idx = len(state_history) - q + i
            if idx < 0:
                result.append(np.zeros(state_dim, dtype=np.float32))
            else:
                result.append(np.array(state_history[idx], dtype=np.float32))
        return result
