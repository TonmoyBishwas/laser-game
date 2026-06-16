"""
Thief — cell-based mover driven by A* pathfinding.

Visible only inside the spotlight; tracks catch-progress when the pointer lingers
on it. Uses pathfinding to navigate to the treasure, then escapes to its spawn door.

Two-phase gameplay:
- PHASE_TREASURE: Navigate from spawn to treasure
- PHASE_ESCAPE: After treasure, navigate back to spawn door (faster)
"""

import math
from enum import auto

import pygame

try:
    from enum import StrEnum
except ImportError:
    # Python < 3.11 fallback
    class StrEnum:
        def _generate_next_value_(name, start, count, last_values):
            return name

from .constants import (
    CATCH_DECAY_RATIO,
    CATCH_HOLD_SECS,
    CATCH_RADIUS,
    CELL_TRANSITION_SECS,
    ESCAPE_SPEED_MULT,
    NO_LASER,
    THIEF_H,
    THIEF_W,
    TREASURE_H,
    TREASURE_W,
)
from .maze import Maze
from .pathfinding import get_next_step

# Phase enum
class Phase(StrEnum):
    TREASURE = auto()  # Going to treasure
    ESCAPE = auto()    # Returning to spawn door

_COLOR_THIEF         = (220, 40, 40)
_COLOR_THIEF_OUTLINE = (140, 0, 0)
# Catch-ring stages. NOTE: the early stage was bright green in the original;
# green is reserved for the laser, so it is cyan here (see ASSET_GUIDE / green rule).
_RING_CHARGE_COLOR   = (80, 200, 255)    # cyan  (was green — green is the laser)
_RING_HALF_COLOR     = (255, 200, 60)    # amber
_RING_FULL_COLOR     = (255, 60, 60)     # red


def _lerp_color(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Thief:
    def __init__(self, maze: Maze, spawn_pos: tuple[int, int] | None = None):
        self.maze = maze
        self.spawn_pos = spawn_pos or maze.spawn
        self.escape_goal: tuple[int, int] | None = None  # Door to escape through
        self.cell: tuple[int, int] = self.spawn_pos
        self.next_cell: tuple[int, int] = self.spawn_pos
        self.transition_t: float = 0.0
        self.pos: list[float] = list(maze.cell_center(*self.spawn_pos))
        self.catch_progress: float = 0.0
        self.reached_treasure: bool = False
        self.escaped: bool = False
        self.caught: bool = False
        self.phase: Phase = Phase.TREASURE
        self.path: list[tuple[int, int]] = []  # Current planned path
        self._replan_needed: bool = True

    # ── Update ────────────────────────────────────────────────────────────────
    def update(
        self,
        dt: float,
        laser_pos: tuple[float, float] | None,
        laser_active: bool,
        laser_cell: tuple[int, int] | None = None,
    ) -> None:
        if self.caught or self.escaped:
            return

        # Determine current speed based on phase
        current_transition_secs = CELL_TRANSITION_SECS * ESCAPE_SPEED_MULT if self.phase == Phase.ESCAPE else CELL_TRANSITION_SECS

        # 1) advance cell-to-cell animation
        if self.cell != self.next_cell:
            self.transition_t = min(1.0, self.transition_t + dt / current_transition_secs)
            sx, sy = self.maze.cell_center(*self.cell)
            tx, ty = self.maze.cell_center(*self.next_cell)
            t = self.transition_t
            self.pos = [sx + (tx - sx) * t, sy + (ty - sy) * t]
            if self.transition_t >= 1.0:
                self.cell = self.next_cell
                self.transition_t = 0.0
                self._check_goal()
                self._replan_needed = True  # Replan at each cell center
        else:
            # at a cell centre — plan next move
            next_step = self._get_next_step(laser_cell)
            if next_step is not None:
                self.next_cell = next_step
                self.transition_t = 0.0
            # else: stay put (at goal or no path)

        # 2) catch-progress
        if laser_active and laser_pos is not None:
            dist = math.hypot(self.pos[0] - laser_pos[0], self.pos[1] - laser_pos[1])
            if dist < CATCH_RADIUS:
                self.catch_progress += dt / CATCH_HOLD_SECS
            else:
                self.catch_progress -= dt / CATCH_HOLD_SECS * CATCH_DECAY_RATIO
        else:
            self.catch_progress -= dt / CATCH_HOLD_SECS * CATCH_DECAY_RATIO

        self.catch_progress = max(0.0, min(1.0, self.catch_progress))
        if self.catch_progress >= 1.0:
            self.caught = True

    def _check_goal(self) -> None:
        """Check if thief reached its current goal."""
        goal = self._get_goal()
        if self.cell == goal:
            if self.phase == Phase.TREASURE:
                # Reached treasure - switch to escape phase
                self.reached_treasure = True
                self.phase = Phase.ESCAPE
                self._replan_needed = True
            elif self.phase == Phase.ESCAPE:
                # Reached door - escaped!
                self.escaped = True

    def _get_goal(self) -> tuple[int, int]:
        """Get the current goal position based on phase."""
        if self.phase == Phase.TREASURE:
            return self.maze.treasure
        else:
            # Use the designated escape door, or fallback to spawn
            return self.escape_goal if self.escape_goal else self.spawn_pos

    def set_escape_goal(self, door_pos: tuple[int, int]) -> None:
        """Set the door position the thief must escape through."""
        self.escape_goal = door_pos

    def _get_next_step(self, laser_cell: tuple[int, int] | None) -> tuple[int, int] | None:
        """Get the next cell to move toward the goal, avoiding the laser."""
        # Use existing path if available and valid
        if not self._replan_needed and self.path and len(self.path) > 1:
            try:
                current_idx = self.path.index(self.cell)
                if current_idx + 1 < len(self.path):
                    return self.path[current_idx + 1]
            except ValueError:
                # Cell not in path, need to replan
                pass

        # Replan path
        goal = self._get_goal()
        if self.cell == goal:
            return None

        # Try with laser avoidance first
        next_step = get_next_step(
            self.cell,
            goal,
            self.maze.is_wall,
            self.maze.in_bounds,
            avoid_pos=laser_cell,
        )

        # Fallback: try without laser avoidance if stuck
        if next_step is None and laser_cell is not None:
            next_step = get_next_step(
                self.cell,
                goal,
                self.maze.is_wall,
                self.maze.in_bounds,
                avoid_pos=None,
            )

        self._replan_needed = False
        return next_step

    def reset(self, new_spawn: tuple[int, int] | None = None) -> None:
        """Reset to initial state. Optionally update spawn position."""
        if new_spawn is not None:
            self.spawn_pos = new_spawn

        self.escape_goal = None
        self.cell = self.spawn_pos
        self.next_cell = self.spawn_pos
        self.transition_t = 0.0
        self.pos = list(self.maze.cell_center(*self.spawn_pos))
        self.catch_progress = 0.0
        self.reached_treasure = False
        self.escaped = False
        self.caught = False
        self.phase = Phase.TREASURE
        self.path = []
        self._replan_needed = True

    # ── Draw ─────────────────────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface) -> None:
        """Render the thief body. Catch ring is drawn separately (above overlay).

        To use art instead of the rectangle: load a PNG with AssetLoader in
        game.py and blit it centered on self.pos here (see games/thief/DESIGN.md).
        """
        rect = pygame.Rect(
            int(self.pos[0] - THIEF_W / 2),
            int(self.pos[1] - THIEF_H / 2),
            THIEF_W,
            THIEF_H,
        )
        pygame.draw.rect(surface, _COLOR_THIEF, rect, border_radius=10)
        pygame.draw.rect(surface, _COLOR_THIEF_OUTLINE, rect, width=3, border_radius=10)

        eye_y = rect.top + THIEF_H // 4
        pygame.draw.circle(surface, (0, 0, 0), (rect.centerx - 10, eye_y), 5)
        pygame.draw.circle(surface, (0, 0, 0), (rect.centerx + 10, eye_y), 5)

    def draw_catch_ring(self, surface: pygame.Surface) -> None:
        """Render the catch-progress ring around the thief (above the overlay)."""
        if self.catch_progress <= 0.0:
            return
        cx, cy = int(self.pos[0]), int(self.pos[1])
        radius = max(THIEF_W, THIEF_H) // 2 + 14

        pygame.draw.circle(surface, (40, 40, 40), (cx, cy), radius, 4)

        if self.catch_progress < 0.5:
            color = _lerp_color(_RING_CHARGE_COLOR, _RING_HALF_COLOR, self.catch_progress * 2)
        else:
            color = _lerp_color(_RING_HALF_COLOR, _RING_FULL_COLOR, (self.catch_progress - 0.5) * 2)

        end_angle = -math.pi / 2 + self.catch_progress * 2 * math.pi
        start_angle = -math.pi / 2
        bbox = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
        pygame.draw.arc(surface, color, bbox, -end_angle, -start_angle, 8)

    def draw_stolen_treasure(self, surface: pygame.Surface, treasure_img: pygame.Surface | None = None) -> None:
        """Draw the stolen treasure on the thief during escape phase."""
        if self.phase != Phase.ESCAPE:
            return

        # Position treasure above the thief's head
        tx = int(self.pos[0])
        ty = int(self.pos[1] - THIEF_H // 2 - TREASURE_H // 2 - 10)

        if treasure_img is not None:
            rect = pygame.Rect(tx - TREASURE_W // 2, ty - TREASURE_H // 2, TREASURE_W, TREASURE_H)
            surface.blit(treasure_img, rect.topleft)
        else:
            # Fallback: draw a simple treasure shape
            rect = pygame.Rect(tx - TREASURE_W // 2, ty - TREASURE_H // 2, TREASURE_W, TREASURE_H)
            color = (255, 215, 0)  # Gold
            pygame.draw.rect(surface, color, rect, border_radius=8)
            pygame.draw.rect(surface, (200, 150, 0), rect, width=2, border_radius=8)
