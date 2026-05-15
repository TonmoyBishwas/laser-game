"""
Train a tabular Q-learning policy for the thief.

Usage
-----
  .\venv\Scripts\python.exe train_thief.py
  .\venv\Scripts\python.exe train_thief.py --episodes 30000 --out policies/thief_qtable.npz

The trained policy is saved to disk as a single .npz; main.py loads it on startup.
"""

import argparse
import math
import random
import time
from collections import deque

from game.maze import Maze
from game.maze_layouts import DEFAULT
from game.rl_env import ALL_MODES, ThiefEnv
from game.rl_policy import QPolicy


def _epsilon_at(ep: int, total: int, end_pct: float, start: float, end: float) -> float:
    """Linear decay from start to end over the first end_pct of training."""
    decay_end = total * end_pct
    if ep >= decay_end:
        return end
    frac = ep / max(1.0, decay_end)
    return start + (end - start) * frac


def train(
    episodes: int,
    out_path: str,
    alpha: float = 0.1,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end:   float = 0.05,
    eps_decay_pct: float = 0.80,
    seed: int = 42,
    log_every: int = 500,
) -> None:
    rng = random.Random(seed)
    maze = Maze(DEFAULT)
    env  = ThiefEnv(maze, seed=seed + 1)
    pol  = QPolicy(alpha=alpha, gamma=gamma, epsilon=eps_start, rng=random.Random(seed + 2))

    print(f"[train] maze signature = {maze.signature}")
    print(f"[train] running {episodes} episodes  alpha={alpha}  gamma={gamma}")
    print(f"[train] eps: {eps_start} -> {eps_end} over {int(eps_decay_pct * 100)}% of training")

    rewards   = deque(maxlen=log_every)
    terminals = deque(maxlen=log_every)
    mode_hits = deque(maxlen=log_every)

    t0 = time.time()
    for ep in range(1, episodes + 1):
        pol.epsilon = _epsilon_at(ep, episodes, eps_decay_pct, eps_start, eps_end)
        mode  = rng.choice(ALL_MODES)
        state = env.reset(mode=mode)
        total_reward = 0.0
        last_terminal = "unknown"

        while True:
            action = pol.act(state)
            next_state, reward, done, info = env.step(action)
            pol.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                last_terminal = info.get("terminal", "unknown")
                break

        rewards.append(total_reward)
        terminals.append(last_terminal)
        mode_hits.append(mode)

        if ep % log_every == 0:
            mean_r  = sum(rewards) / len(rewards)
            win_pct = 100.0 * sum(1 for t in terminals if t == "treasure") / len(terminals)
            caught_pct = 100.0 * sum(1 for t in terminals if t == "caught") / len(terminals)
            timeout_pct = 100.0 * sum(1 for t in terminals if t == "timeout") / len(terminals)
            elapsed = time.time() - t0
            print(
                f"[train] ep={ep:6d}  eps={pol.epsilon:.3f}  "
                f"mean_R={mean_r:+.3f}  win={win_pct:5.1f}%  "
                f"caught={caught_pct:5.1f}%  timeout={timeout_pct:5.1f}%  "
                f"states={len(pol.q):,}  elapsed={elapsed:5.1f}s"
            )

    win_rate    = sum(1 for t in terminals if t == "treasure") / max(1, len(terminals))
    caught_rate = sum(1 for t in terminals if t == "caught")   / max(1, len(terminals))

    pol.metadata = {
        "episodes":       episodes,
        "alpha":          alpha,
        "gamma":          gamma,
        "eps_start":      eps_start,
        "eps_end":        eps_end,
        "eps_decay_pct":  eps_decay_pct,
        "maze_signature": maze.signature,
        "final_win_rate": float(win_rate),
        "final_caught_rate": float(caught_rate),
        "trained_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    pol.epsilon = 0.0
    pol.save(out_path)
    print(f"[train] done. final win_rate={win_rate:.2%}  caught_rate={caught_rate:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,   default=20000)
    parser.add_argument("--alpha",    type=float, default=0.1)
    parser.add_argument("--gamma",    type=float, default=0.95)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out",      type=str,   default="policies/thief_qtable.npz")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        out_path=args.out,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
