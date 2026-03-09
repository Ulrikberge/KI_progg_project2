"""
NeuralNetworkManager (NNM): owns and trains the three MuZero networks.

  NNr (representation):  φ = [s_{k-q}, ..., s_k]  →  σ_k  (abstract state)
  NNd (dynamics):        (σ, a)                    →  (σ', r_predicted)
  NNp (prediction):      σ                          →  (π, v)

Training uses BPTT via JAX automatic differentiation over a composite loss
that jointly optimises all three networks.
"""

import pickle
import jax
import jax.numpy as jnp
import optax

import config
from muzero.neural_network import NeuralNetwork


class NeuralNetworkManager:

    def __init__(self, state_dim: int, action_count: int, rng: jax.Array):
        self.state_dim = state_dim
        self.action_count = action_count

        # Build the three networks
        repr_input_dim = config.LOOKBACK * state_dim
        dyn_input_dim  = config.ABSTRACT_STATE_DIM + action_count
        pred_input_dim = config.ABSTRACT_STATE_DIM

        self.nn_r = NeuralNetwork(config.REPR_HIDDEN_SIZES, config.ABSTRACT_STATE_DIM, "tanh")
        self.nn_d = NeuralNetwork(config.DYN_HIDDEN_SIZES,  config.ABSTRACT_STATE_DIM + 1, "linear")
        self.nn_p = NeuralNetwork(config.PRED_HIDDEN_SIZES,  action_count + 1, "linear")

        # Initialise parameters
        rng, r1, r2, r3 = jax.random.split(rng, 4)
        self.params_r = self.nn_r.init_params(r1, (repr_input_dim,))
        self.params_d = self.nn_d.init_params(r2, (dyn_input_dim,))
        self.params_p = self.nn_p.init_params(r3, (pred_input_dim,))

        # Optimisers — Adam with global gradient-norm clipping to prevent explosion
        _opt = lambda: optax.chain(
            optax.clip_by_global_norm(config.GRAD_CLIP_NORM),
            optax.adam(config.LEARNING_RATE),
        )
        self.opt_r = _opt()
        self.opt_d = _opt()
        self.opt_p = _opt()

        self.opt_state_r = self.opt_r.init(self.params_r)
        self.opt_state_d = self.opt_d.init(self.params_d)
        self.opt_state_p = self.opt_p.init(self.params_p)

        # JIT-compile frequently used ops
        self._represent_jit  = jax.jit(self._represent_fn)
        self._dynamics_jit   = jax.jit(self._dynamics_fn)
        self._predict_jit    = jax.jit(self._predict_fn)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _represent_fn(self, params_r, state_seq_flat: jnp.ndarray) -> jnp.ndarray:
        return self.nn_r.forward(params_r, state_seq_flat)

    def _dynamics_fn(self, params_d, abstract_state: jnp.ndarray,
                     action_onehot: jnp.ndarray):
        inp = jnp.concatenate([abstract_state, action_onehot], axis=-1)
        out = self.nn_d.forward(params_d, inp)
        next_abstract = jnp.tanh(out[..., :-1])
        # Bound reward to [-REWARD_SCALE, REWARD_SCALE] — prevents OOD abstract
        # states from producing arbitrarily large rewards in MCTS rollout.
        reward = jnp.tanh(out[..., -1]) * config.REWARD_SCALE
        return next_abstract, reward

    def _predict_fn(self, params_p, abstract_state: jnp.ndarray):
        out = self.nn_p.forward(params_p, abstract_state)
        policy_logits = out[..., :-1]
        # Bound value to [-VALUE_SCALE, VALUE_SCALE] — same reason as above.
        value = jnp.tanh(out[..., -1]) * config.VALUE_SCALE
        return policy_logits, value

    def represent(self, state_seq: list) -> jnp.ndarray:
        """Map a sequence of q game states to an abstract state σ."""
        flat = jnp.array([s for s in state_seq]).flatten()
        return self._represent_jit(self.params_r, flat)

    def dynamics(self, abstract_state: jnp.ndarray, action: int):
        """Map (σ, a) → (σ', predicted_reward)."""
        onehot = jax.nn.one_hot(action, self.action_count)
        return self._dynamics_jit(self.params_d, abstract_state, onehot)

    def predict(self, abstract_state: jnp.ndarray):
        """Map σ → (policy_logits, value)."""
        return self._predict_jit(self.params_p, abstract_state)

    # ------------------------------------------------------------------
    # Training (BPTT through composite network)
    # ------------------------------------------------------------------

    def _single_sample_loss(self, params_r, params_d, params_p,
                             state_seq, actions_seq, pi_seq, v_seq, r_seq):
        """
        Loss for one training sample.
        Uses jax.lax.scan over the w roll-ahead steps so JAX sees the full
        computation graph without Python-level unrolling.
        """
        sigma = self._represent_fn(params_r, state_seq)

        # Clip value targets to prevent enormous MSE losses early in training
        v_seq = jnp.clip(v_seq, -config.VALUE_CLIP, config.VALUE_CLIP)

        def rollout_step(sigma, inputs):
            a, pi_t, v_t, r_t = inputs
            pi_logits, v_pred = self._predict_fn(params_p, sigma)

            policy_loss = -jnp.sum(pi_t * jax.nn.log_softmax(pi_logits))
            value_loss  = config.VALUE_LOSS_COEFF  * (v_pred - v_t) ** 2

            a_onehot = jax.nn.one_hot(a, self.action_count)
            sigma_next, r_pred = self._dynamics_fn(params_d, sigma, a_onehot)
            reward_loss = config.REWARD_LOSS_COEFF * (r_pred - r_t) ** 2

            step_loss = policy_loss + value_loss + reward_loss
            return sigma_next, step_loss

        _, step_losses = jax.lax.scan(
            rollout_step, sigma, (actions_seq, pi_seq, v_seq, r_seq)
        )
        return jnp.mean(step_losses)

    def _loss_fn(self, params_r, params_d, params_p, batch):
        """
        Composite loss over a minibatch — vectorised with jax.vmap.
        batch = (state_seqs, actions, pi_targets, v_targets, r_targets)
          shapes: (mbs, q*state_dim), (mbs,w), (mbs,w,n_actions), (mbs,w), (mbs,w)
        """
        state_seqs, actions, pi_targets, v_targets, r_targets = batch

        per_sample = jax.vmap(
            lambda ss, a, pi, v, r: self._single_sample_loss(
                params_r, params_d, params_p, ss, a, pi, v, r
            )
        )(state_seqs, actions, pi_targets, v_targets, r_targets)

        return jnp.mean(per_sample)

    def train(self, episode_buffer, mbs: int) -> float:
        """Sample a minibatch from the buffer and do one gradient step."""
        batch = episode_buffer.sample_minibatch(
            mbs, config.LOOKBACK, config.ROLL_AHEAD, self.state_dim
        )
        if batch is None:
            return 0.0

        # Convert to JAX arrays once
        batch = tuple(jnp.array(b) for b in batch)

        grad_fn = jax.value_and_grad(self._loss_fn, argnums=(0, 1, 2))
        loss, (gr, gd, gp) = grad_fn(
            self.params_r, self.params_d, self.params_p, batch
        )

        updates_r, self.opt_state_r = self.opt_r.update(gr, self.opt_state_r)
        updates_d, self.opt_state_d = self.opt_d.update(gd, self.opt_state_d)
        updates_p, self.opt_state_p = self.opt_p.update(gp, self.opt_state_p)

        self.params_r = optax.apply_updates(self.params_r, updates_r)
        self.params_d = optax.apply_updates(self.params_d, updates_d)
        self.params_p = optax.apply_updates(self.params_p, updates_p)

        return float(loss)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_params(self, path: str):
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"r": self.params_r, "d": self.params_d, "p": self.params_p}, f)

    def load_params(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.params_r = data["r"]
        self.params_d = data["d"]
        self.params_p = data["p"]
