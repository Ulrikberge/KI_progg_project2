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
from games.bitfall_game import GameStateManager
from muzero.neural_network_manager import NeuralNetworkManager
from muzero.rl_manager import RLManager
from visualization.visualizer import Visualizer


def main():
    print("=" * 60)
    print(f"MuZero Knockoff  |  BitFall  {config.GRID_ROWS}x{config.GRID_COLS}")
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
    viz.save_metrics("results/training_metrics.png")

    # --- Demo A: actor-only (NNr + NNp, no MCTS) — satisfies spec actor requirement ---
    print("\nRunning actor-only demo (NNr + NNp, no MCTS) ...")
    _actor_demo(gsm, rlm)

    # --- Demo B: full MCTS demo with game rendering ---
    print("\nRunning MCTS demo episode ...")
    _demo_episode(gsm, rlm)
    viz.save_game_gif("results/demo_episode.gif")

    gsm.close()
    viz.close()


def _demo_episode(gsm, rlm: RLManager):
    """Play one episode greedily using the trained policy, with game rendering."""
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
        rlm.visualizer.render_game_state(state)
        if done:
            break

    cfg.VISUALIZE = False
    print(f"Demo episode reward: {total_reward:.1f}")


def _actor_demo(gsm, rlm: RLManager):
    """
    Play one episode using only NNr + NNp (no MCTS) — the 'actor' described in the spec.
    Action = argmax of NNp policy logits applied to the current abstract state.
    """
    import config as cfg
    import jax

    state = gsm.reset()
    total_reward = 0.0
    state_history = [state]

    for _ in range(cfg.STEPS_PER_EPISODE):
        if gsm.is_terminal(state):
            break
        phi_k = rlm._build_lookback(state_history, cfg.LOOKBACK)
        sigma_k = rlm.asm.get_abstract_state(phi_k)
        policy_logits, _ = rlm.asm.get_policy_and_value(sigma_k)
        probs = jax.nn.softmax(policy_logits)
        action = int(np.argmax(probs))
        state, reward, done = gsm.step(state, action)
        total_reward += reward
        state_history.append(state)
        if done:
            break

    print(f"Actor-only demo reward: {total_reward:.1f}")


if __name__ == "__main__":
    main()
