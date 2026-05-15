"""
Sparse tabular Q-learning policy.

State  : tuple of ints (thief_r, thief_c, laser_r, laser_c)
Action : 0..3  (N/E/S/W)
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
        # metadata — set by trainer and consumed by main.py for HUD
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
        # break ties randomly so behaviour isn't stuck in one corner
        best = np.flatnonzero(v == v.max())
        return int(self.rng.choice(best.tolist()))

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        q_sa = self._get(state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(self._get(next_state).max())
        q_sa[action] += self.alpha * (target - q_sa[action])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.q:
            states = np.zeros((0, 4), dtype=np.int16)
            values = np.zeros((0, NUM_ACTIONS), dtype=np.float32)
        else:
            states = np.array(list(self.q.keys()), dtype=np.int16)
            values = np.stack(list(self.q.values())).astype(np.float32)

        # metadata stored as keyword args, all strings or floats
        meta = {f"meta_{k}": np.asarray(v) for k, v in self.metadata.items()}
        np.savez_compressed(path, states=states, values=values, **meta)
        print(f"[policy] saved {len(self.q)} states -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "QPolicy":
        data = np.load(path, allow_pickle=False)
        policy = cls(epsilon=0.0)  # greedy at runtime
        states = data["states"]
        values = data["values"]
        for s, v in zip(states.tolist(), values):
            policy.q[tuple(s)] = v.astype(np.float32)

        # decode metadata
        for k in data.files:
            if k.startswith("meta_"):
                raw = data[k]
                policy.metadata[k[5:]] = raw.item() if raw.ndim == 0 else raw.tolist()

        print(f"[policy] loaded {len(policy.q)} states from {path}")
        return policy
