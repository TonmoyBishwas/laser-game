"""
Hand-designed and procedurally generated maze layouts.

Cell encoding:
  0 = open path
  1 = wall
  'S' = thief spawn (single, open)
  'T' = treasure (single, open)

Layout is rows x cols. (0,0) is top-left in game coordinates.

NOTE: The DEFAULT layout uses a pre-trained Q-table (qtable.npz). Random layouts
use A* pathfinding instead and don't depend on the Q-table.
"""

from typing import Any

# Try to import the random maze generator
try:
    from .maze_gen import get_random_layout

    _has_random = True
except ImportError:
    _has_random = False

# 15 cols x 9 rows.
# Two main vertical corridors at col 1 (left) and col 13 (right).
# Horizontal corridors at rows 2, 6 connect them fully.
# Row 4 has partial corridors on each side (detours but no central crossing).
# This biases routes toward the screen edges where the real detector works best.
DEFAULT = [
    [1, 1, 1, 1, 1, 1, 1, 'T', 1, 1, 1, 1, 1, 1, 1],   # row 0
    [1, 0, 1, 1, 1, 1, 1,  0,  1, 1, 1, 1, 1, 0, 1],   # row 1
    [1, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 1],   # row 2 — top corridor
    [1, 0, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 0, 1],   # row 3
    [1, 0, 0, 0, 0, 0, 1,  1,  1, 0, 0, 0, 0, 0, 1],   # row 4 — partial mid corridor
    [1, 0, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 0, 1],   # row 5
    [1, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 1],   # row 6 — lower corridor
    [1, 0, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 0, 1],   # row 7
    [1, 0, 0, 0, 0, 0, 0, 'S', 0, 0, 0, 0, 0, 0, 1],   # row 8 — spawn row
]


def random_layout(seed: int | None = None) -> list[list[Any]]:
    """
    Generate a random maze layout.

    Args:
        seed: Random seed for reproducibility. None for fully random.

    Returns:
        A GRID_ROWS x GRID_COLS maze grid with 'S', 'T', 0, and 1 values.
        Guaranteed to have at least 2 disjoint paths from spawn to treasure.

    Note: Requires pathfinding-based AI (not the Q-table).
    """
    if not _has_random:
        raise RuntimeError("Random maze generation requires maze_gen module")
    return get_random_layout(seed=seed)
