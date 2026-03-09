# =============================================================================
# MuZero Knockoff — Central Configuration
# All pivotal parameters live here. Change them in ONE PLACE.
# =============================================================================

# --- Game: BitFall ---
# Grid: GRID_ROWS rows of falling debris + 1 receptor row at the bottom.
# Actions: 0 = slide left, 1 = stay, 2 = slide right.
GRID_ROWS = 6                    # number of debris rows visible
GRID_COLS = 8                    # width of the grid
DEBRIS_DENSITY = 0.3             # probability a cell is debris in each new top row

# --- Episode loop ---
N_EPISODES = 200                 # Ne: total training episodes
STEPS_PER_EPISODE = 200          # Nes: steps per episode (BitFall has no terminal)
TRAINING_INTERVAL = 10           # It: train Psi every this many episodes

# --- u-MCTS ---
N_MCTS_SIMULATIONS = 50          # Ms: simulations per move
MAX_ROLLOUT_DEPTH = 5            # dmax: max depth in DO_ROLLOUT

# --- Training ---
MINIBATCH_SIZE = 32              # mbs
LOOKBACK = 4                     # q: how many past states feed into NNr
ROLL_AHEAD = 5                   # w: how many future steps for BPTT targets
DISCOUNT_GAMMA = 0.99            # discount factor for value targets
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 1.0             # gradient clipping (global norm)

# Output scaling for bounded NN heads (tanh × scale)
# NNd reward output:  tanh(x) * REWARD_SCALE  → [-REWARD_SCALE, REWARD_SCALE]
# NNp value  output:  tanh(x) * VALUE_SCALE   → [-VALUE_SCALE, VALUE_SCALE]
# BitFall: max reward per step ≈ GRID_COLS (all debris caught), so REWARD_SCALE = GRID_COLS.
# Episode return ≈ STEPS_PER_EPISODE * avg_reward_per_step.
REWARD_SCALE      = 8.0          # ≈ GRID_COLS (max debris that can be caught per step)
VALUE_SCALE       = 50.0         # ≈ expected episode return with reasonable play
VALUE_CLIP        = 200.0        # clip MC-return value targets

# MCTS exploration: Dirichlet noise at root prevents early over-commitment to
# one action when Q-values are biased by random weight initialisation.
DIRICHLET_ALPHA   = 0.3          # concentration (lower = more noise)
DIRICHLET_EPSILON = 0.25         # mixing weight for noise vs prior

# Loss coefficients (from MuZero paper)
VALUE_LOSS_COEFF  = 0.25
REWARD_LOSS_COEFF = 1.0

# --- Neural Networks ---
ABSTRACT_STATE_DIM = 64          # dimensionality of abstract (hidden) states

# NNr (representation): input = q * state_dim, output = ABSTRACT_STATE_DIM
REPR_HIDDEN_SIZES = [128, 128]

# NNd (dynamics): input = ABSTRACT_STATE_DIM + n_actions, output = ABSTRACT_STATE_DIM + 1
DYN_HIDDEN_SIZES = [128, 128]

# NNp (prediction): input = ABSTRACT_STATE_DIM, output = n_actions + 1
PRED_HIDDEN_SIZES = [64, 64]

# --- UCB exploration constant (for tree policy) ---
UCB_C = 1.25

# --- Visualization ---
VISUALIZE = False                # Set True to render game states during play
PLOT_METRICS = True              # Plot reward/loss curves during training

# --- Misc ---
RANDOM_SEED = 42
SAVE_PARAMS_PATH = "checkpoints/muzero_params.pkl"
