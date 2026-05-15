"""
Gym-style environment used by train_thief.py.

State  : (thief_r, thief_c, laser_r, laser_c)   laser_(-1,-1) = no laser
Action : 0 N | 1 E | 2 S | 3 W
Reward : per-step shaping + terminal rewards
"""

import math
import random

from game.maze import Maze, ACTION_VECS, GRID_COLS, GRID_ROWS

# Reward shaping
R_STEP_OPEN    = -0.01   # discourage wandering
R_STEP_WALL    = -0.05   # discourage hitting walls (still valid action)
R_CAUGHT       = -1.00   # terminal
R_TREASURE     =  1.00   # terminal
R_TIMEOUT      = -0.50   # terminal — too slow

# In-sim "time" — same scale as the runtime CELL_TRANSITION_SECS.
DT_SIM = 0.4    # seconds per step (matches thief cell-to-cell pace)

# Episode cap
MAX_STEPS = 250

# Laser modes
MODE_IDLE       = "idle"
MODE_PATROL     = "patrol"
MODE_TRACKING   = "tracking"
MODE_RANDOM     = "random"
ALL_MODES = [MODE_IDLE, MODE_PATROL, MODE_TRACKING, MODE_RANDOM]

NO_LASER = (-1, -1)


class LaserSim:
    """Generates a laser-cell trajectory for one training episode."""

    def __init__(self, maze: Maze, mode: str, rng: random.Random):
        self.maze = maze
        self.mode = mode
        self.rng = rng
        self.t = 0

        # Pick parameters for this episode
        if mode == MODE_PATROL:
            # Patrol along col=1 or col=13, top-to-bottom-cycling
            self.patrol_col = rng.choice([1, GRID_COLS - 2])
            self.patrol_dir = rng.choice([-1, 1])
            self.patrol_r   = rng.randint(0, GRID_ROWS - 1)
        elif mode == MODE_TRACKING:
            # Laser tracks the thief with lag + jitter
            self.lag = rng.uniform(1.5, 3.5)   # in steps
            self.jitter = rng.uniform(0.1, 0.3)
            self.cell = (rng.randint(0, GRID_ROWS - 1), rng.randint(0, GRID_COLS - 1))
        elif mode == MODE_RANDOM:
            self.cell = self._random_open_cell()
            self.persistence = rng.randint(2, 8)
        # MODE_IDLE: nothing

    def _random_open_cell(self) -> tuple[int, int]:
        cells = list(self.maze.all_open_cells())
        return self.rng.choice(cells)

    def next_cell(self, thief_cell: tuple[int, int]) -> tuple[int, int]:
        """Return the laser's cell for this simulation step."""
        self.t += 1

        if self.mode == MODE_IDLE:
            return NO_LASER

        if self.mode == MODE_PATROL:
            self.patrol_r = (self.patrol_r + self.patrol_dir) % GRID_ROWS
            return (self.patrol_r, self.patrol_col)

        if self.mode == MODE_RANDOM:
            self.persistence -= 1
            if self.persistence <= 0:
                self.cell = self._random_open_cell()
                self.persistence = self.rng.randint(2, 8)
            return self.cell

        # MODE_TRACKING — drift toward thief with jitter
        tr, tc = thief_cell
        cr, cc = self.cell
        # smooth move 1/lag toward thief
        cr = cr + (tr - cr) / max(self.lag, 1.0)
        cc = cc + (tc - cc) / max(self.lag, 1.0)
        # jitter
        cr += self.rng.uniform(-self.jitter, self.jitter)
        cc += self.rng.uniform(-self.jitter, self.jitter)
        # round to integer cell, clamp
        r = max(0, min(GRID_ROWS - 1, int(round(cr))))
        c = max(0, min(GRID_COLS - 1, int(round(cc))))
        self.cell = (cr, cc)
        return (r, c)


class ThiefEnv:
    """One thief, one maze, one laser simulator. Episode-based."""

    def __init__(self, maze: Maze, seed: int | None = None):
        self.maze = maze
        self.rng = random.Random(seed)

        # Catch-hold logic: how many consecutive steps the laser has been on the thief
        self._catch_streak = 0
        self._catch_threshold = max(1, int(round(1.0 / DT_SIM)))   # 1.0 s of holding

        # Episode state
        self.thief_cell: tuple[int, int] = maze.spawn
        self.laser_sim: LaserSim | None = None
        self.steps = 0
        self.done = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, mode: str | None = None) -> tuple:
        if mode is None:
            mode = self.rng.choice(ALL_MODES)
        self.laser_sim = LaserSim(self.maze, mode, self.rng)
        self.thief_cell = self.maze.spawn
        self._catch_streak = 0
        self.steps = 0
        self.done = False
        return self._state(NO_LASER)

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        if self.done:
            raise RuntimeError("Cannot step a done env; call reset()")

        # 1) move the thief
        new_r, new_c, hit_wall = self.maze.step(*self.thief_cell, action)
        self.thief_cell = (new_r, new_c)

        # 2) advance laser
        laser_cell = self.laser_sim.next_cell(self.thief_cell)

        # 3) compute reward + check terminals
        reward = R_STEP_WALL if hit_wall else R_STEP_OPEN
        info = {"hit_wall": hit_wall, "laser_cell": laser_cell, "mode": self.laser_sim.mode}

        # Caught? — laser at the same cell as thief
        if laser_cell != NO_LASER and laser_cell == self.thief_cell:
            self._catch_streak += 1
        else:
            self._catch_streak = 0

        if self._catch_streak >= self._catch_threshold:
            reward += R_CAUGHT
            self.done = True
            info["terminal"] = "caught"
        elif self.thief_cell == self.maze.treasure:
            reward += R_TREASURE
            self.done = True
            info["terminal"] = "treasure"

        self.steps += 1
        if not self.done and self.steps >= MAX_STEPS:
            reward += R_TIMEOUT
            self.done = True
            info["terminal"] = "timeout"

        return self._state(laser_cell), reward, self.done, info

    def _state(self, laser_cell: tuple[int, int]) -> tuple[int, int, int, int]:
        tr, tc = self.thief_cell
        lr, lc = laser_cell
        return (tr, tc, lr, lc)


# ── Convenience ─────────────────────────────────────────────────────────────

def cells_within_radius(maze: Maze, center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = center
    out = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if maze.is_wall(r, c):
                continue
            if math.hypot(r - cr, c - cc) <= radius:
                out.append((r, c))
    return out
