"""
Maze grid + lookup helpers.

The maze is a list of rows, each row a list of cell values (see maze_layouts.py).
Game-space pixel coordinates are computed from CELL_W/CELL_H in config.
"""

import hashlib
import json
from typing import Iterable

from game.config import CELL_H, CELL_W, GAME_H, GAME_W, GRID_COLS, GRID_ROWS

# Action constants (used by RL + thief)
N, E, S, W = 0, 1, 2, 3
ACTION_VECS = {
    N: (-1, 0),   # row delta, col delta
    E: ( 0, 1),
    S: ( 1, 0),
    W: ( 0,-1),
}
ACTION_NAMES = {N: "N", E: "E", S: "S", W: "W"}


class Maze:
    def __init__(self, grid: list[list]):
        if len(grid) != GRID_ROWS or any(len(r) != GRID_COLS for r in grid):
            raise ValueError(
                f"Maze grid must be {GRID_ROWS} rows × {GRID_COLS} cols, "
                f"got {len(grid)} × {len(grid[0]) if grid else 0}"
            )
        self.grid = grid
        self.spawn:    tuple[int, int] | None = None
        self.treasure: tuple[int, int] | None = None
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if v == 'S':
                    self.spawn = (r, c)
                elif v == 'T':
                    self.treasure = (r, c)
        if self.spawn is None:
            raise ValueError("Maze has no 'S' (spawn) cell")
        if self.treasure is None:
            raise ValueError("Maze has no 'T' (treasure) cell")

    # ── Cell queries ──────────────────────────────────────────────────────────

    def is_wall(self, r: int, c: int) -> bool:
        if r < 0 or r >= GRID_ROWS or c < 0 or c >= GRID_COLS:
            return True
        return self.grid[r][c] == 1

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS

    def neighbours(self, r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in ACTION_VECS.values():
            nr, nc = r + dr, c + dc
            if not self.is_wall(nr, nc):
                out.append((nr, nc))
        return out

    def step(self, r: int, c: int, action: int) -> tuple[int, int, bool]:
        """Apply action. Returns (new_r, new_c, hit_wall)."""
        dr, dc = ACTION_VECS[action]
        nr, nc = r + dr, c + dc
        if self.is_wall(nr, nc):
            return r, c, True
        return nr, nc, False

    # ── Coordinate conversions ────────────────────────────────────────────────

    @staticmethod
    def cell_center(r: int, c: int) -> tuple[float, float]:
        return (c + 0.5) * CELL_W, (r + 0.5) * CELL_H

    @staticmethod
    def pixel_to_cell(x: float, y: float) -> tuple[int, int]:
        c = int(x // CELL_W)
        r = int(y // CELL_H)
        c = max(0, min(GRID_COLS - 1, c))
        r = max(0, min(GRID_ROWS - 1, r))
        return r, c

    # ── Iteration ─────────────────────────────────────────────────────────────

    def all_open_cells(self) -> Iterable[tuple[int, int]]:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if not self.is_wall(r, c):
                    yield r, c

    # ── Identity hash (so policy NPZ can be matched to the maze) ──────────────

    @property
    def signature(self) -> str:
        flat = [[1 if self.is_wall(r, c) else 0 for c in range(GRID_COLS)] for r in range(GRID_ROWS)]
        flat[self.spawn[0]][self.spawn[1]] = 'S'
        flat[self.treasure[0]][self.treasure[1]] = 'T'
        blob = json.dumps(flat, separators=(",", ":")).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
