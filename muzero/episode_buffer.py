"""
EpisodeBuffer (EB): stores completed episodes and samples minibatches for BPTT.

Each episode is a list of timestep tuples:
    (s_k, v*_k, π_k, a_{k+1}, r*_{k+1})

The buffer can fetch chunks of state-action-reward-policy-value data
that the NNM uses as BPTT targets.
"""

import random
import numpy as np


class EpisodeBuffer:

    def __init__(self):
        self.episodes: list[list] = []   # list of episodes; each episode is a list of steps

    def add_episode(self, episode: list):
        """
        Append a completed episode.
        episode: list of (state, value, policy, action, reward) tuples.
        """
        if len(episode) > 0:
            self.episodes.append(episode)

    def __len__(self):
        return len(self.episodes)

    def sample_minibatch(self, mbs: int, q: int, w: int, state_dim: int):
        """
        Sample mbs training examples.
        Each example is a (q lookback states, w actions, w policies, w values, w rewards) tuple.

        Returns:
          state_seqs  — (mbs, q * state_dim)  flattened state sequences
          actions     — (mbs, w)
          pi_targets  — (mbs, w, n_actions)
          v_targets   — (mbs, w)
          r_targets   — (mbs, w)
        or None if not enough data.
        """
        if len(self.episodes) == 0:
            return None

        # Collect examples
        state_seqs, actions_list, pi_list, v_list, r_list = [], [], [], [], []

        for _ in range(mbs):
            ep = random.choice(self.episodes)
            if len(ep) < w + 1:
                continue

            # Pick a random start timestep with enough look-ahead
            k = random.randint(0, len(ep) - w - 1)

            # Build lookback window φ_k = [s_{k-q}, ..., s_k]
            seq = []
            for i in range(q):
                idx = k - q + 1 + i
                if idx < 0:
                    seq.append(np.zeros(state_dim, dtype=np.float32))
                else:
                    seq.append(np.array(ep[idx][0], dtype=np.float32))
            state_seqs.append(np.concatenate(seq))  # shape (q * state_dim,)

            # Roll-ahead targets
            ep_actions, ep_pi, ep_v, ep_r = [], [], [], []
            n_actions = len(ep[k][2])
            for t in range(w):
                step = ep[min(k + t, len(ep) - 1)]
                ep_actions.append(step[3])             # action
                ep_pi.append(np.array(step[2], dtype=np.float32))  # policy
                ep_v.append(float(step[1]))            # value
                ep_r.append(float(step[4]))            # reward

            actions_list.append(ep_actions)
            pi_list.append(ep_pi)
            v_list.append(ep_v)
            r_list.append(ep_r)

        if len(state_seqs) == 0:
            return None

        return (
            np.array(state_seqs, dtype=np.float32),           # (mbs, q*state_dim)
            np.array(actions_list, dtype=np.int32),            # (mbs, w)
            np.array(pi_list, dtype=np.float32),               # (mbs, w, n_actions)
            np.array(v_list, dtype=np.float32),                # (mbs, w)
            np.array(r_list, dtype=np.float32),                # (mbs, w)
        )
