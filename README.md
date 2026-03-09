# MuZero Knockoff — IT-3105 AI Programming, Spring 2025

Implementation of the MuZero game-playing algorithm from scratch, applied to a
Gymnasium environment (default: CartPole-v1).
Built with **JAX + Flax + Optax**.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

All parameters live in `config.py` — change the game, network sizes, training
schedule, etc. in **one place**.

---

## Project Structure

```
project2/
├── config.py                   ← ALL parameters here (one place)
├── main.py                     ← entry point
├── requirements.txt
│
├── games/                      ← SimWorld (game-specific code only)
│   └── gym_game.py             │  VideoGameSimulator + GameStateManager
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
    └── visualizer.py           ← reward/loss plots + game-state rendering
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

## Key Design Decisions

### SimWorld / AI Separation (critical for grading)
- `games/` = SimWorld. Contains ALL game-specific logic.
- `muzero/` = AI. **Never imports from `games/`.**
- The `GameStateManager` is passed into `RLManager` as a dependency in `main.py`.

### All Parameters in One Place
See `config.py`. Changing the Gymnasium environment requires editing only
`GYM_ENV_ID`. Changing the game changes nothing in `muzero/`.

### Numerical Stability
Several issues were encountered and fixed during development:

| Problem | Fix |
|---------|-----|
| Exploding loss (1.9 → 4000+) | `optax.clip_by_global_norm` on all optimizers |
| OOD abstract states in rollout producing huge rewards | `tanh × REWARD_SCALE` and `tanh × VALUE_SCALE` on NNd/NNp outputs |
| Python loops inside `jax.grad` | Rewrote BPTT with `jax.lax.scan` + `jax.vmap` |
| MCTS biased to one action by random Q-values | Min-max Q normalisation within siblings (MuZero paper, Appendix B) |
| Noisy MCTS policy preventing exploration | Dirichlet noise on root priors (α=0.3, ε=0.25) |
| Noisy MCTS Q-values as value targets | Replaced with actual Monte Carlo episode returns |

---

## Current State (as of hand-off)

### What works
- Full EPISODE_LOOP running end-to-end without errors
- Loss is **stable** (no longer diverging): converges to ~5–20 range
- Dirichlet noise + Q-normalization prevents rewards from going *below* random play
- All 8 required components (VideoGameSimulator, GSM, ASM, NN, NNM, u-MCTS, EB, RLM) implemented
- Clean SimWorld / AI separation passes the critical divide requirement
- Params checkpoint saved to `checkpoints/muzero_params.pkl`

### What doesn't work yet
- **Policy is not actually learning** to play CartPole better over 200 episodes.
  Rewards hover around 8–25 (random play baseline is ~20).
  The fundamental reason: MuZero requires an accurate NNr to map game states to
  meaningful abstract states. With random NNr, NNd/NNp have no structured
  representation to work from. The bootstrapping loop (MCTS → training → MCTS)
  needs thousands of episodes to converge.

### Root cause (for the video)
MuZero has a bootstrapping chicken-and-egg problem:
- Good MCTS requires accurate NNd/NNp
- Accurate NNd/NNp require good training data
- Good training data requires good MCTS
The original paper trained on **millions** of frames; we have 200 episodes on a CPU.

---

## What to Do Next

### Priority 1 — More episodes
Change `config.py`:
```python
N_EPISODES = 2000
N_MCTS_SIMULATIONS = 20   # reduce to keep it fast
```
Running overnight should show clear learning. CartPole is "solved" (~500 reward)
by most deep RL methods in ~1000 episodes with a direct NN, but MuZero's
additional indirection needs more.

### Priority 2 — Better game choice
CartPole is challenging for MuZero because:
- Reward = 1.0 every step regardless of action quality (no differential signal)
- NNd has to learn continuous physics from scratch

Better options:
- **BitFall** (described in the task PDF): discrete grid, clear reward differences
  per action, low branching factor. Much easier for NNd to learn.
- **MountainCar** or **Acrobot**: negative reward per step → clearer signal

### Priority 3 — Curriculum / warm-up
Add a supervised pre-training phase for NNr using random-play data before starting
MCTS training. This gives NNr a structured representation before the bootstrapping
loop begins.

### Priority 4 — Video deliverable
The video is the only deliverable (4 points). Required content:
1. Explain RL, model-based vs model-free, MCTS, MuZero (1 pt)
2. Describe your implementation in detail — use the architecture diagram above (2 pts)
3. Show the video game briefly (0 pts if missing, -1 pt if absent)
4. Show results — even if they are not great, explain WHY (1 pt)
All diagrams must be your own (not from lecture notes/papers).
Max 10 minutes. Deadline: **Noon, Friday May 2, 2025**.

---

## Switching Games

To switch to a different Gymnasium environment, change **only** `config.py`:
```python
GYM_ENV_ID = "LunarLander-v2"  # or "MountainCar-v0", etc.
```
The entire `muzero/` stack is game-agnostic. You may need to adjust
`VALUE_SCALE` and `REWARD_SCALE` to match the new environment's reward range.

To implement **BitFall** (or another custom game), write a new file in `games/`
that implements the same interface as `gym_game.py`:
```python
class VideoGameSimulator:
    def reset(self) -> state: ...
    def step(self, state, action) -> (next_state, reward, done): ...

class GameStateManager:
    def get_legal_actions(self, state) -> list[int]: ...
    def is_terminal(self, state) -> bool: ...
    def get_state_shape(self) -> tuple: ...
    def get_action_count(self) -> int: ...
```
Then update the import in `main.py`. No changes needed in `muzero/`.
