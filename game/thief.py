"""
Thief — cell-based mover driven by a Q-policy.

Visible only inside the spotlight; tracks catch-progress when the laser lingers
on it. When at a cell centre, it consults the policy for the next action and
animates a smooth slide to the neighbouring cell over CELL_TRANSITION_SECS.
"""

import math

import pygame

from game.config import (
    CATCH_DECAY_RATIO,
    CATCH_HOLD_SECS,
    CATCH_RADIUS,
    CELL_TRANSITION_SECS,
    THIEF_H,
    THIEF_W,
)
from game.maze import ACTION_VECS, Maze
from game.rl_env import NO_LASER
from game.rl_policy import QPolicy

_COLOR_THIEF        = (220, 40, 40)
_COLOR_THIEF_OUTLINE = (140, 0, 0)
_RING_CHARGE_COLOR  = (80, 255, 80)
_RING_HALF_COLOR    = (255, 200, 60)
_RING_FULL_COLOR    = (255, 60, 60)


def _lerp_color(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Thief:
    def __init__(self, maze: Maze, policy: QPolicy):
        self.maze = maze
        self.policy = policy
        self.cell: tuple[int, int] = maze.spawn        # current cell (after arrival)
        self.next_cell: tuple[int, int] = maze.spawn   # target cell (where we're animating to)
        self.transition_t: float = 0.0                 # 0..1 along the current animation
        self.pos: list[float] = list(maze.cell_center(*maze.spawn))
        self.catch_progress: float = 0.0
        self.reached: bool = False
        self.caught: bool = False

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        dt: float,
        laser_pos: tuple[float, float] | None,
        laser_active: bool,
        laser_cell: tuple[int, int] | None = None,
    ) -> None:
        if self.reached or self.caught:
            return

        # 1) advance cell-to-cell animation
        if self.cell != self.next_cell:
            self.transition_t = min(1.0, self.transition_t + dt / CELL_TRANSITION_SECS)
            sx, sy = self.maze.cell_center(*self.cell)
            tx, ty = self.maze.cell_center(*self.next_cell)
            t = self.transition_t
            self.pos = [sx + (tx - sx) * t, sy + (ty - sy) * t]
            if self.transition_t >= 1.0:
                self.cell = self.next_cell
                self.transition_t = 0.0
                if self.cell == self.maze.treasure:
                    self.reached = True
                    return
        else:
            # at a cell centre — pick the next move from the policy
            state = (
                self.cell[0],
                self.cell[1],
                laser_cell[0] if laser_cell is not None else NO_LASER[0],
                laser_cell[1] if laser_cell is not None else NO_LASER[1],
            )
            action = self.policy.act(state, greedy=True)
            self.last_state = state
            self.last_action = action
            dr, dc = ACTION_VECS[action]
            nr, nc = self.cell[0] + dr, self.cell[1] + dc
            if not self.maze.is_wall(nr, nc):
                self.next_cell = (nr, nc)
                self.transition_t = 0.0
            # else: stay (wall) — policy will be queried again next frame

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

    def reset(self) -> None:
        self.cell = self.maze.spawn
        self.next_cell = self.maze.spawn
        self.transition_t = 0.0
        self.pos = list(self.maze.cell_center(*self.maze.spawn))
        self.catch_progress = 0.0
        self.reached = False
        self.caught = False

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        """Render the thief body. Catch ring is drawn separately (above overlay)."""
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

    def draw_silhouette(self, surface: pygame.Surface) -> None:
        """Debug-only: faint thief outline visible even in the dark."""
        rect = pygame.Rect(
            int(self.pos[0] - THIEF_W / 2),
            int(self.pos[1] - THIEF_H / 2),
            THIEF_W,
            THIEF_H,
        )
        sil = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(sil, (220, 60, 60, 60), sil.get_rect(), border_radius=10)
        pygame.draw.rect(sil, (220, 60, 60, 120), sil.get_rect(), width=2, border_radius=10)
        surface.blit(sil, rect.topleft)

    def draw_catch_ring(self, surface: pygame.Surface) -> None:
        """Render the catch-progress ring around the thief (drawn above the overlay)."""
        if self.catch_progress <= 0.0:
            return
        cx, cy = int(self.pos[0]), int(self.pos[1])
        radius = max(THIEF_W, THIEF_H) // 2 + 14

        # Background circle
        pygame.draw.circle(surface, (40, 40, 40), (cx, cy), radius, 4)

        # Fill arc
        if self.catch_progress < 0.5:
            color = _lerp_color(_RING_CHARGE_COLOR, _RING_HALF_COLOR, self.catch_progress * 2)
        else:
            color = _lerp_color(_RING_HALF_COLOR, _RING_FULL_COLOR, (self.catch_progress - 0.5) * 2)

        end_angle = -math.pi / 2 + self.catch_progress * 2 * math.pi
        start_angle = -math.pi / 2
        # pygame.draw.arc draws from start_angle CCW to end_angle, but our visual is CW;
        # flip by using (-end_angle, -start_angle).
        bbox = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
        pygame.draw.arc(surface, color, bbox, -end_angle, -start_angle, 8)
