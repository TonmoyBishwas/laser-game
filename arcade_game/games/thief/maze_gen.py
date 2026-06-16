"""
Random maze generation for the thief game.

Generates random 15x9 mazes with guaranteed multiple paths between
spawn and treasure locations.
"""

import random
from collections import deque
from typing import Any

from .constants import GRID_COLS, GRID_ROWS

# Cell values
WALL = 1
OPEN = 0
SPAWN = 'S'
TREASURE = 'T'
DOOR = 'D'  # Escape door


def generate_maze(
    seed: int | None = None,
    min_path_count: int = 2,
    min_distance: int = 6,
    wall_density: float = 0.35,
    num_doors: int = 4,
) -> list[list[Any]]:
    """
    Generate a random maze with guaranteed multiple paths.

    Args:
        seed: Random seed for reproducibility. None for random.
        min_path_count: Minimum number of disjoint paths from spawn to treasure
        min_distance: Minimum Manhattan distance between spawn and treasure
        wall_density: Approximate ratio of wall cells (0.0 to 1.0)
        num_doors: Number of escape doors to place

    Returns:
        A GRID_ROWS x GRID_COLS maze grid with 'S', 'T', 'D', 0, and 1 values.
    """
    if seed is not None:
        random.seed(seed)

    # Start with all open
    grid = [[OPEN for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    # Add outer walls (border)
    for r in range(GRID_ROWS):
        grid[r][0] = WALL
        grid[r][GRID_COLS - 1] = WALL
    for c in range(GRID_COLS):
        grid[0][c] = WALL
        grid[GRID_ROWS - 1][c] = WALL

    # Try to generate a valid maze
    max_attempts = 100
    for attempt in range(max_attempts):
        maze = [row[:] for row in grid]  # Copy base grid

        # Add random walls
        open_cells = [
            (r, c)
            for r in range(1, GRID_ROWS - 1)
            for c in range(1, GRID_COLS - 1)
        ]
        random.shuffle(open_cells)

        wall_count = int(len(open_cells) * wall_density)
        for i in range(wall_count):
            r, c = open_cells[i]
            maze[r][c] = WALL

        # Pick spawn and treasure positions
        available = [(r, c) for r in range(1, GRID_ROWS - 1)
                      for c in range(1, GRID_COLS - 1)
                      if maze[r][c] == OPEN]

        if len(available) < 2:
            continue

        spawn, treasure = random.sample(available, 2)

        # Check minimum distance
        if abs(spawn[0] - treasure[0]) + abs(spawn[1] - treasure[1]) < min_distance:
            continue

        maze[spawn[0]][spawn[1]] = SPAWN
        maze[treasure[0]][treasure[1]] = TREASURE

        # Add multiple escape doors
        available_for_doors = [(r, c) for r in range(1, GRID_ROWS - 1)
                               for c in range(1, GRID_COLS - 1)
                               if maze[r][c] == OPEN and (r, c) != spawn and (r, c) != treasure]

        doors_to_add = min(num_doors, len(available_for_doors))
        if doors_to_add > 0:
            door_positions = random.sample(available_for_doors, doors_to_add)
            for dr, dc in door_positions:
                maze[dr][dc] = DOOR

        # Check path count
        paths = _count_disjoint_paths(maze, spawn, treasure, min_path_count)
        if paths >= min_path_count:
            return maze

    # Fallback: return a simple maze with guaranteed paths
    return _fallback_maze(num_doors=num_doors)


def _count_disjoint_paths(
    maze: list[list[Any]],
    spawn: tuple[int, int],
    treasure: tuple[int, int],
    target: int,
) -> int:
    """
    Count disjoint paths from spawn to treasure using BFS.

    Two paths are disjoint if they don't share any cells (except endpoints).
    We count by repeatedly removing the found path and searching again.
    """
    count = 0
    blocked = set()  # Cells that are already used by previous paths

    for _ in range(target):
        path = _find_shortest_path(maze, spawn, treasure, blocked)
        if not path:
            break

        # Mark this path's cells as blocked (except endpoints)
        for cell in path[1:-1]:
            blocked.add(cell)

        count += 1

    return count


def _find_shortest_path(
    maze: list[list[Any]],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """BFS to find shortest path avoiding blocked cells."""
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path

        r, c = current
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if (nr, nc) in visited:
                continue
            if (nr, nc) in blocked and (nr, nc) != goal:
                continue
            if nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS:
                continue
            if maze[nr][nc] == WALL:
                continue

            visited.add((nr, nc))
            queue.append(((nr, nc), path + [(nr, nc)]))

    return None


def _fallback_maze(num_doors: int = 4) -> list[list[Any]]:
    """
    Simple fallback maze with guaranteed multiple paths.
    Used if random generation fails.
    """
    maze = [[OPEN for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    # Add some walls to create interesting paths
    wall_pattern = [
        [3, 6, 9],  # Rows with extra walls
        [2, 5, 8, 11],  # Columns with gaps
    ]

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Outer walls
            if r == 0 or r == GRID_ROWS - 1 or c == 0 or c == GRID_COLS - 1:
                maze[r][c] = WALL
            # Pattern walls
            elif r in wall_pattern[0] and c not in wall_pattern[1]:
                maze[r][c] = WALL

    # Spawn at bottom, treasure at top
    spawn = (GRID_ROWS - 2, GRID_COLS // 2)
    treasure = (1, GRID_COLS // 2)
    maze[spawn[0]][spawn[1]] = SPAWN
    maze[treasure[0]][treasure[1]] = TREASURE

    # Add doors at corners
    door_positions = [
        (1, 2), (1, GRID_COLS - 3),
        (GRID_ROWS - 2, 2), (GRID_ROWS - 2, GRID_COLS - 3)
    ]
    for i, (dr, dc) in enumerate(door_positions[:num_doors]):
        if maze[dr][dc] == OPEN:
            maze[dr][dc] = DOOR

    return maze


def get_random_layout(seed: int | None = None) -> list[list[Any]]:
    """Convenience function to get a random maze layout."""
    return generate_maze(seed=seed)
