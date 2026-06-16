"""
A* pathfinding for the thief.

Replaces the Q-table policy to support random mazes and dynamic goals.
The thief can avoid the laser pointer by penalizing cells near it.
"""

import heapq
from typing import Callable

from .maze import ACTION_VECS, N, E, S, W

# Heuristic: Manhattan distance
def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    is_wall: Callable[[int, int], bool],
    in_bounds: Callable[[int, int], bool],
    avoid_pos: tuple[int, int] | None = None,
    avoid_radius: int = 2,
) -> list[tuple[int, int]]:
    """
    Find a path from start to goal using A*.

    Args:
        start: (row, col) starting position
        goal: (row, col) target position
        is_wall: function(r, c) -> bool, returns True if cell is a wall
        in_bounds: function(r, c) -> bool, returns True if cell is in grid
        avoid_pos: (row, col) of laser pointer to avoid, or None
        avoid_radius: how many cells away from avoid_pos to penalize

    Returns:
        List of (row, col) cells from start to goal (inclusive).
        Empty list if no path exists (shouldn't happen with valid mazes).
    """
    if start == goal:
        return [start]

    # Priority queue: (f_score, g_score, current_cell, path)
    # f_score = g_score + heuristic
    pq = [(0, 0, start, [start])]
    visited = {start}

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current == goal:
            return path

        # Explore neighbors
        for action in (N, E, S, W):
            dr, dc = ACTION_VECS[action]
            nr, nc = current[0] + dr, current[1] + dc

            if not in_bounds(nr, nc) or is_wall(nr, nc):
                continue

            next_cell = (nr, nc)
            if next_cell in visited:
                continue

            # Calculate movement cost
            step_cost = 1

            # Penalize cells near the laser
            if avoid_pos is not None:
                dist_to_laser = _manhattan(next_cell, avoid_pos)
                if dist_to_laser <= avoid_radius:
                    # Higher penalty for closer cells
                    step_cost += (avoid_radius - dist_to_laser + 1) * 3

            new_g = g + step_cost
            new_f = new_g + _manhattan(next_cell, goal)

            new_path = path + [next_cell]
            heapq.heappush(pq, (new_f, new_g, next_cell, new_path))
            visited.add(next_cell)

    # No path found (shouldn't happen with valid maze generation)
    return []


def get_next_step(
    current: tuple[int, int],
    goal: tuple[int, int],
    is_wall: Callable[[int, int], bool],
    in_bounds: Callable[[int, int], bool],
    avoid_pos: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """
    Get the next step toward the goal.

    Convenience wrapper for find_path that returns just the next cell.
    Returns None if already at goal or no path exists.
    """
    if current == goal:
        return None

    path = find_path(current, goal, is_wall, in_bounds, avoid_pos)
    if len(path) < 2:
        return None

    return path[1]  # path[0] is current, path[1] is next step
