"""
Visualizer: game-state rendering and training metric plots.
Respects config.VISUALIZE and config.PLOT_METRICS flags.

Saving:
  save_metrics(path)   — write the training metrics plot as a PNG
  save_game_gif(path)  — write all frames collected via render_game_state as a GIF
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import config


class Visualizer:

    def __init__(self):
        self._metrics_fig = None
        self._ax_reward = None
        self._ax_loss = None
        self._game_fig = None
        self._game_ax = None
        self._game_im = None
        self._frames = []          # RGB frames collected for GIF export
        plt.ion()   # interactive mode — non-blocking updates

    # ------------------------------------------------------------------
    # Game state rendering
    # ------------------------------------------------------------------

    def render_game_state(self, state: np.ndarray):
        """
        Render the BitFall grid as a coloured matplotlib image.
        Debris rows = red, receptor row = blue, empty = white.
        Always collects the frame for GIF export.
        Live display only when config.VISUALIZE is True.
        """
        rows = config.GRID_ROWS
        cols = config.GRID_COLS
        debris = state[: rows * cols].reshape(rows, cols)
        receptor = state[rows * cols :]

        # Build an RGB image
        grid = np.ones((rows + 1, cols, 3), dtype=np.float32)  # white bg
        for r in range(rows):
            for c in range(cols):
                if debris[r, c]:
                    grid[r, c] = [0.9, 0.2, 0.2]   # red = debris
        for c in range(cols):
            if receptor[c]:
                grid[rows, c] = [0.2, 0.4, 0.9]    # blue = receptor

        # Always keep frame for GIF
        self._frames.append(grid.copy())

        if not config.VISUALIZE:
            return

        if self._game_fig is None:
            self._game_fig, self._game_ax = plt.subplots(figsize=(4, 4))
            self._game_ax.set_title("BitFall")
            self._game_ax.set_xticks([])
            self._game_ax.set_yticks([])
            self._game_im = self._game_ax.imshow(grid, vmin=0, vmax=1)
        else:
            self._game_im.set_data(grid)

        plt.pause(0.05)

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

    # ------------------------------------------------------------------
    # Save to files
    # ------------------------------------------------------------------

    def save_metrics(self, path: str = "results/training_metrics.png"):
        """Save the training metrics figure to a PNG file."""
        if self._metrics_fig is None:
            print("No metrics figure to save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._metrics_fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Metrics plot saved to {path}")

    def save_game_gif(self, path: str = "results/demo_episode.gif", fps: int = 10):
        """Save collected game frames as an animated GIF."""
        if not self._frames:
            print("No game frames to save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("BitFall — demo episode")
        im = ax.imshow(self._frames[0], vmin=0, vmax=1)

        def update(frame):
            im.set_data(frame)
            return [im]

        anim = animation.FuncAnimation(
            fig, update, frames=self._frames,
            interval=1000 // fps, blit=True
        )
        anim.save(path, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"Demo GIF saved to {path}")

    def close(self):
        plt.ioff()
        plt.close("all")
