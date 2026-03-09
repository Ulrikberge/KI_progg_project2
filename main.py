"""
main.py — Entry point for the MuZero Knockoff.

1. Reads all parameters from config.py (change things there, not here).
2. Instantiates SimWorld (GameStateManager) and all AI components.
3. Runs EPISODE_LOOP via RLManager.
4. Optionally saves trained parameters and plays one demo episode.
"""

import jax
import numpy as np

import config
from games.gym_game import GameStateManager
from muzero.neural_network_manager import NeuralNetworkManager
from muzero.rl_manager import RLManager
from visualization.visualizer import Visualizer


def main():
    print("=" * 60)
    print(f"MuZero Knockoff  |  env: {config.GYM_ENV_ID}")
    print(f"Episodes: {config.N_EPISODES}  |  MCTS sims: {config.N_MCTS_SIMULATIONS}")
    print("=" * 60)

    # Seeded RNG
    rng = jax.random.PRNGKey(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # --- SimWorld (game-specific) ---
    gsm = GameStateManager()
    state_dim = int(np.prod(gsm.get_state_shape()))
    action_count = gsm.get_action_count()
    print(f"State dim: {state_dim}  |  Actions: {action_count}")

    # --- AI components (game-agnostic) ---
    nnm = NeuralNetworkManager(state_dim, action_count, rng)
    viz = Visualizer()

    rlm = RLManager(gsm, nnm, visualizer=viz)

    # --- Train ---
    trained_nnm = rlm.run()
    trained_nnm.save_params(config.SAVE_PARAMS_PATH)
    print(f"\nTraining complete. Params saved to {config.SAVE_PARAMS_PATH}")

    # --- Demo: play one episode with visualisation on ---
    print("\nRunning demo episode (visualisation on) ...")
    _demo_episode(gsm, rlm, viz)

    gsm.close()
    viz.close()


def _demo_episode(gsm, rlm: RLManager, viz: Visualizer):
    """Play one episode greedily using the trained policy."""
    import config as cfg
    cfg.VISUALIZE = True

    state = gsm.reset()
    total_reward = 0.0
    state_history = [state]

    for k in range(cfg.STEPS_PER_EPISODE):
        if gsm.is_terminal(state):
            break
        phi_k = rlm._build_lookback(state_history, cfg.LOOKBACK)
        sigma_k = rlm.asm.get_abstract_state(phi_k)
        pi_k, _ = rlm.mcts.search(sigma_k, cfg.N_MCTS_SIMULATIONS, cfg.MAX_ROLLOUT_DEPTH)
        action = int(np.argmax(pi_k))   # greedy
        state, reward, done = gsm.step(state, action)
        total_reward += reward
        state_history.append(state)
        viz.render_game_state(state)
        if done:
            break

    print(f"Demo episode reward: {total_reward:.1f}")
    cfg.VISUALIZE = False


if __name__ == "__main__":
    main()
