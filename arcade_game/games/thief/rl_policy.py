"""
Sparse tabular Q-learning policy (the thief's pre-trained brain).

State  : tuple of ints (thief_r, thief_c, laser_r, laser_c)
Action : 0..3  (N/E/S/W)

At runtime we only LOAD a trained table (qtable.npz) and read the best action.
Training happens in the separate real project; do not retrain here.
"""

import random
from pathlib import Path

import numpy as np

NUM_ACTIONS = 4


class QPolicy:
    def __init__(
        self,
        alpha:   float = 0.1,
        gamma:   float = 0.95,
        epsilon: float = 1.0,
        rng:     random.Random | None = None,
    ):
        self.q: dict[tuple, np.ndarray] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.metadata: dict = {}

    # ── Lookup ────────────────────────────────────────────────────────────────
    def _get(self, state: tuple) -> np.ndarray:
        v = self.q.get(state)
        if v is None:
            v = np.zeros(NUM_ACTIONS, dtype=np.float32)
            self.q[state] = v
        return v

    def values(self, state: tuple) -> np.ndarray:
        """Read-only view of Q-values for a state (without inserting)."""
        v = self.q.get(state)
        return v if v is not None else np.zeros(NUM_ACTIONS, dtype=np.float32)

    # ── Acting ────────────────────────────────────────────────────────────────
    def act(self, state: tuple, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randint(0, NUM_ACTIONS - 1)
        v = self._get(state)
        best = np.flatnonzero(v == v.max())   # break ties randomly
        return int(self.rng.choice(best.tolist()))

    # ── Persistence ───────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path) -> "QPolicy":
        data = np.load(path, allow_pickle=False)
        policy = cls(epsilon=0.0)  # greedy at runtime
        states = data["states"]
        values = data["values"]
        for s, v in zip(states.tolist(), values):
            policy.q[tuple(s)] = v.astype(np.float32)

        for k in data.files:
            if k.startswith("meta_"):
                raw = data[k]
                policy.metadata[k[5:]] = raw.item() if raw.ndim == 0 else raw.tolist()

        print(f"[policy] loaded {len(policy.q)} states from {path}")
        return policy
