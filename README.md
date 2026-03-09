# MuZero Knockoff — IT-3105 AI Programming, Spring 2025

Implementation of the MuZero game-playing algorithm from scratch, applied to **BitFall**.
Built with **JAX + Flax + Optax**.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

All parameters live in `config.py` — change network sizes, training schedule, grid dimensions, etc. in **one place**.

---

## Project Structure

```
project2/
├── config.py                   ← ALL parameters here (one place)
├── main.py                     ← entry point
├── requirements.txt
│
├── games/                      ← SimWorld (game-specific code only)
│   ├── bitfall_game.py         │  BitFall: VideoGameSimulator + GameStateManager
│   └── gym_game.py             │  (kept for reference — original CartPole wrapper)
│
├── muzero/                     ← AI system (zero game-specific imports)
│   ├── neural_network.py       │  Flax MLP wrapper
│   ├── neural_network_manager.py  NNr / NNd / NNp + BPTT training
│   ├── abstract_state_manager.py  Interface between MCTS and NNM
│   ├── mcts.py                 │  u-MCTS (UCB, expand, rollout, backprop)
│   ├── episode_buffer.py       │  Episode storage + minibatch sampling
│   └── rl_manager.py           │  EPISODE_LOOP orchestrator
│
└── visualization/
    └── visualizer.py           ← live reward/loss plots + BitFall grid rendering
```

---

## Architecture Overview

MuZero learns by interacting with three coupled neural networks (Ψ):

| Network | Input | Output | Role |
|---------|-------|--------|------|
| **NNr** (representation) | q real game states | abstract state σ | Encodes game history |
| **NNd** (dynamics) | (σ, action) | (σ′, r̂) | Predicts next abstract state + reward |
| **NNp** (prediction) | σ | (π, v̂) | Predicts policy + value |

The training loop (`EPISODE_LOOP` in `rl_manager.py`):
1. Reset the game to initial state s₀
2. For each timestep k:
   - Build φₖ = [s_{k-q}, …, sₖ] (lookback window)
   - σₖ = NNr(φₖ)
   - Run u-MCTS (Ms simulations) → πₖ, v*ₖ
   - Sample action aₖ₊₁ from πₖ
   - Step the real game → sₖ₊₁, rₖ₊₁
3. After episode: compute actual MC returns as value targets v*ₖ
4. Every `TRAINING_INTERVAL` episodes: train Ψ via BPTT

---

## BitFall — Game Description

BitFall is a discrete grid game described in the assignment spec.

- **Grid**: `GRID_ROWS` rows of falling debris above a receptor row (`GRID_COLS` wide).
- **Debris**: Each timestep, the bottom row is evaluated and removed; all rows shift down; a new random row is added at the top.
- **Receptor**: A row of blue segments at the bottom. Can slide left, stay, or slide right each timestep. Wraps around the edges.
- **Scoring** per segment pair at the bottom:
  - Receptor completely covers debris with excess → **+debris_size** (positive)
  - Debris completely covers receptor with excess → **−receptor_size** (negative)
  - Partial overlap or equal size → 0
- **No terminal state** — episodes run for exactly `STEPS_PER_EPISODE` steps.

**State representation**: flat float32 array of length `(GRID_ROWS + 1) × GRID_COLS`.
First `GRID_ROWS × GRID_COLS` values = debris grid (row-major); last `GRID_COLS` values = receptor row. All binary (0/1).

**Why BitFall over CartPole**: discrete grid makes NNd much easier to learn; clear per-step reward differences give meaningful signal from episode 1; low branching factor (3 actions) suits MCTS.

---

## Key Design Decisions

### SimWorld / AI Separation (critical for grading)
- `games/` = SimWorld. Contains ALL game-specific logic.
- `muzero/` = AI. **Never imports from `games/`.**
- The `GameStateManager` is passed into `RLManager` as a dependency in `main.py`.

### All Parameters in One Place
See `config.py`. Switching the game requires changing only the import in `main.py` and adjusting `REWARD_SCALE` / `VALUE_SCALE`. Nothing in `muzero/` changes.

### Numerical Stability
Several issues were encountered and fixed during development:

| Problem | Fix |
|---------|-----|
| Exploding loss (1.9 → 4000+) | `optax.clip_by_global_norm` on all optimizers |
| OOD abstract states producing unbounded rewards | `tanh × REWARD_SCALE` and `tanh × VALUE_SCALE` on NNd/NNp outputs |
| Python loops inside `jax.grad` | Rewrote BPTT with `jax.lax.scan` + `jax.vmap` |
| MCTS locked onto one action by random Q-values | Min-max Q normalisation within siblings (MuZero paper, Appendix B) |
| Noisy MCTS policy preventing exploration | Dirichlet noise on root priors (α=0.3, ε=0.25) |
| Noisy MCTS Q-values as value targets | Replaced with actual Monte Carlo episode returns |

---

## Current State

### What works
- Full EPISODE_LOOP running end-to-end on BitFall without errors
- Loss **decreases ~10× over 200 episodes**: 825 → 84 (stable, no divergence)
- Rewards consistently **above the random-play baseline** (~100–130): agent scores 150–200 per episode
- All 8 required components implemented: VideoGameSimulator, GSM, ASM, NN, NNM, u-MCTS, EB, RLM
- Clean SimWorld / AI separation passes the critical divide requirement
- BitFall grid renders as a colour grid during the post-training demo episode
- Live reward/loss plot window updates during training (`PLOT_METRICS = True`)
- Params checkpoint saved to `checkpoints/muzero_params.pkl`

### What doesn't work yet
- **Reward curve is flat** — rewards don't trend upward over episodes.
  The networks are fitting (loss falls) but the MCTS policy isn't improving yet.
  Root cause: MuZero's bootstrapping loop needs many more episodes before NNr
  learns a structured representation that NNd/NNp can exploit.

### Root cause (for the video)
MuZero has a bootstrapping chicken-and-egg problem:
- Good MCTS requires accurate NNd/NNp
- Accurate NNd/NNp require good training data
- Good training data requires good MCTS

The original paper trained on **millions** of frames; we have 200 episodes on a CPU.
The loss decrease confirms the networks are learning; the flat reward confirms the
bootstrap loop hasn't closed yet.

---

## What to Do Next

### Priority 1 — More episodes (most impactful)
```python
# config.py
N_EPISODES         = 1000
N_MCTS_SIMULATIONS = 20    # reduce to keep runtime manageable
STEPS_PER_EPISODE  = 100
```
Running overnight should show a reward trend upward. With 1000 episodes the bootstrap
loop has a real chance to close.

### Priority 2 — Video deliverable
The video is the **only deliverable** (4 points). Required content:
1. Explain RL, model-based vs model-free, MCTS, MuZero (1 pt)
2. Describe your implementation in detail — use the architecture table above (2 pts)
3. Show the video game briefly — use the demo episode render (0 pts if missing)
4. Show results — loss curve + reward curve; explain the bootstrapping problem (1 pt)

All diagrams must be your own (not from lecture notes/papers).
Max 10 minutes. Deadline: **Noon, Friday May 2, 2025**.

### Priority 3 — Visualisation for the video
`config.PLOT_METRICS = True` produces a live matplotlib window with reward and loss curves.
After training, the demo episode renders the BitFall grid (red = debris, blue = receptor).
Screenshot or screen-record these for the video.

### Priority 4 — Curriculum / warm-up (optional improvement)
Add a supervised pre-training phase for NNr using random-play data before starting
MCTS training. This gives NNr a structured representation before the bootstrapping
loop begins, potentially accelerating convergence.

---

## Switching Games

To switch back to a Gymnasium environment, change the import in `main.py`:
```python
from games.gym_game import GameStateManager
```
and restore `GYM_ENV_ID` in `config.py`. No changes needed in `muzero/`.

To add another custom game, implement the same interface in `games/`:
```python
class VideoGameSimulator:
    def reset(self) -> np.ndarray: ...
    def step(self, state, action) -> (next_state, reward, done): ...

class GameStateManager:
    def reset(self) -> np.ndarray: ...
    def step(self, state, action) -> (next_state, reward, done): ...
    def get_legal_actions(self, state) -> list[int]: ...
    def is_terminal(self, state) -> bool: ...
    def get_state_shape(self) -> tuple: ...
    def get_action_count(self) -> int: ...
    def close(self): ...
```
Then update the import in `main.py`. No changes needed in `muzero/`.
