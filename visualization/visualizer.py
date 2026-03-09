"""
Visualizer: game-state rendering and training metric plots.
Respects config.VISUALIZE and config.PLOT_METRICS flags.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import config


class Visualizer:

    def __init__(self):
        self._metrics_fig = None
        self._ax_reward = None
        self._ax_loss = None
        plt.ion()   # interactive mode — non-blocking updates

    # ------------------------------------------------------------------
    # Game state rendering
    # ------------------------------------------------------------------

    def render_game_state(self, state: np.ndarray):
        """
        Display a game state observation.
        For vector states (e.g. CartPole) we print to stdout.
        Override this method for pixel-based environments.
        """
        if not config.VISUALIZE:
            return
        print(f"  state: {np.round(state, 3)}")

    # ------------------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------------------

    def plot_training_metrics(self, episode: int, rewards: list, losses: list):
        """Live-update reward and loss curves."""
        if not config.PLOT_METRICS:
            return

        if self._metrics_fig is None:
            self._metrics_fig, (self._ax_reward, self._ax_loss) = plt.subplots(
                1, 2, figsize=(10, 4)
            )
            self._metrics_fig.suptitle("MuZero Training Metrics")

        self._ax_reward.cla()
        self._ax_reward.plot(rewards, color="steelblue")
        self._ax_reward.set_title("Episode Reward")
        self._ax_reward.set_xlabel("Episode")
        self._ax_reward.set_ylabel("Total Reward")

        self._ax_loss.cla()
        if losses:
            xs = [i * config.TRAINING_INTERVAL for i in range(1, len(losses) + 1)]
            self._ax_loss.plot(xs, losses, color="tomato")
        self._ax_loss.set_title("Training Loss")
        self._ax_loss.set_xlabel("Episode")
        self._ax_loss.set_ylabel("Loss")

        self._metrics_fig.tight_layout()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close("all")
