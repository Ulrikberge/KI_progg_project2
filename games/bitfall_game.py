"""
SimWorld: BitFall game implementation.
This module is the ONLY place where game-specific logic lives.
The muzero/ package must never import from here directly.

BitFall rules (from assignment spec):
  - Debris (red) falls straight down, one row per timestep.
  - Receptors (blue) occupy the bottom row and can move left, right, or stay.
  - Each timestep:
      1. Compare bottom debris row with receptor row → reward.
      2. Remove bottom debris row; shift all debris down one row.
      3. Add a new random debris row at the top.
      4. Move receptor according to action.
  - Receptor wraps around the edges (np.roll).
  - Scoring per segment pair:
      * Receptor completely covers debris with excess → +debris_size (positive)
      * Debris completely covers receptor with excess → -receptor_size (negative)
      * Equal size / partial overlap → 0
  - No terminal state; episodes run for exactly STEPS_PER_EPISODE steps.
"""

import numpy as np
import config


class VideoGameSimulator:
    """
    Maps (state, action) → (next_state, reward, done).
    State is a flat float32 array: debris grid rows followed by receptor row.
    """

    def __init__(self):
        self._debris = None   # shape (GRID_ROWS, GRID_COLS)
        self._receptor = None  # shape (GRID_COLS,)

    # ------------------------------------------------------------------
    # Public interface (same as gym_game.py)
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        rows, cols = config.GRID_ROWS, config.GRID_COLS
        self._debris = (
            np.random.random((rows, cols)) < config.DEBRIS_DENSITY
        ).astype(np.float32)
        # Start receptor as a centred block of width cols//2
        self._receptor = np.zeros(cols, dtype=np.float32)
        half = cols // 2
        start = (cols - half) // 2
        self._receptor[start: start + half] = 1.0
        return self._get_state()

    def step(self, state: np.ndarray, action: int):
        """
        Execute one BitFall timestep.
        Actions: 0 = left, 1 = stay, 2 = right.
        Returns (next_state, reward, done).  done is always False.
        """
        # 1. Score: compare bottom debris row against receptor
        reward = _score_rows(self._debris[-1], self._receptor)

        # 2. Shift debris down (drop bottom row, add new row at top)
        new_top = (
            np.random.random(config.GRID_COLS) < config.DEBRIS_DENSITY
        ).astype(np.float32)
        self._debris = np.vstack([new_top, self._debris[:-1]])

        # 3. Move receptor
        if action == 0:        # left
            self._receptor = np.roll(self._receptor, -1)
        elif action == 2:      # right
            self._receptor = np.roll(self._receptor, +1)
        # action == 1 → stay

        return self._get_state(), float(reward), False

    def close(self):
        pass

    @property
    def state_shape(self) -> tuple:
        return ((config.GRID_ROWS + 1) * config.GRID_COLS,)

    @property
    def action_count(self) -> int:
        return 3  # left, stay, right

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """Flatten debris grid + receptor row into a single 1D float32 array."""
        return np.concatenate(
            [self._debris.flatten(), self._receptor], axis=0
        ).astype(np.float32)


class GameStateManager:
    """
    Understands game states and provides the interface the AI uses.
    Never imported by muzero/ — passed in as a dependency.
    """

    def __init__(self):
        self._sim = VideoGameSimulator()

    def reset(self) -> np.ndarray:
        return self._sim.reset()

    def step(self, state: np.ndarray, action: int):
        """Returns (next_state, reward, done)."""
        return self._sim.step(state, action)

    def get_legal_actions(self, state: np.ndarray) -> list:
        return list(range(self._sim.action_count))

    def is_terminal(self, state: np.ndarray) -> bool:
        return False  # BitFall has no terminal state

    def get_state_shape(self) -> tuple:
        return self._sim.state_shape

    def get_action_count(self) -> int:
        return self._sim.action_count

    def close(self):
        self._sim.close()


# ----------------------------------------------------------------------
# Reward calculation (module-level helper, no game state)
# ----------------------------------------------------------------------

def _find_segments(row: np.ndarray) -> list:
    """
    Find contiguous 1-segments in a binary row.
    Returns list of (start_inclusive, end_exclusive) tuples.
    """
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(row):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(row)))
    return segments


def _score_rows(debris_row: np.ndarray, receptor_row: np.ndarray) -> float:
    """
    Compare bottom debris row against receptor row and return reward.

    For each (receptor_seg, debris_seg) pair that overlaps:
      - receptor completely covers debris with excess  → +debris_size
      - debris completely covers receptor with excess  → -receptor_size
      - otherwise (partial overlap, equal size/pos)   → 0
    """
    d_segs = _find_segments(debris_row)
    r_segs = _find_segments(receptor_row)
    reward = 0.0
    for rs, re in r_segs:
        r_size = re - rs
        for ds, de in d_segs:
            d_size = de - ds
            # No overlap at all
            if re <= ds or de <= rs:
                continue
            if rs <= ds and re >= de and r_size > d_size:
                # Receptor completely covers debris with excess → positive
                reward += d_size
            elif ds <= rs and de >= re and d_size > r_size:
                # Debris completely covers receptor with excess → negative
                reward -= r_size
            # else: partial overlap or same size → 0
    return reward
