"""
SimWorld: Gymnasium-based game implementation.
This module is the ONLY place where game-specific logic lives.
The muzero/ package must never import from here directly — it interacts
only through the interfaces defined by VideoGameSimulator and GameStateManager.
"""

import numpy as np
import gymnasium as gym
import config


class VideoGameSimulator:
    """
    Maps (state, action) → (next_state, reward, done).
    Wraps a Gymnasium environment as the low-level simulator.
    """

    def __init__(self):
        self._env = gym.make(config.GYM_ENV_ID)
        self._current_state = None

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=None)
        self._current_state = obs
        return obs

    def step(self, state: np.ndarray, action: int):
        """
        Execute action from the given state.
        Returns (next_state, reward, done).
        Note: Gymnasium envs are stateful, so 'state' is used for documentation
        purposes; the env advances from its internal current state.
        """
        obs, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        self._current_state = obs
        return obs, float(reward), done

    def render(self):
        """Render the environment (only when VISUALIZE is True)."""
        if config.VISUALIZE:
            self._env.render()

    def close(self):
        self._env.close()

    @property
    def state_shape(self) -> tuple:
        return self._env.observation_space.shape

    @property
    def action_count(self) -> int:
        return int(self._env.action_space.n)


class GameStateManager:
    """
    Understands game states and provides the interface the AI uses.
    Never imported by muzero/ — passed in as a dependency.
    """

    def __init__(self):
        self._sim = VideoGameSimulator()
        self._done = False

    def reset(self) -> np.ndarray:
        self._done = False
        return self._sim.reset()

    def step(self, state: np.ndarray, action: int):
        """Returns (next_state, reward, done)."""
        next_state, reward, done = self._sim.step(state, action)
        self._done = done
        return next_state, reward, done

    def get_legal_actions(self, state: np.ndarray) -> list:
        """All discrete actions are always legal in Gymnasium envs."""
        return list(range(self._sim.action_count))

    def is_terminal(self, state: np.ndarray) -> bool:
        return self._done

    def get_state_shape(self) -> tuple:
        return self._sim.state_shape

    def get_action_count(self) -> int:
        return self._sim.action_count

    def render(self):
        self._sim.render()

    def close(self):
        self._sim.close()
