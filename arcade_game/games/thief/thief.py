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
    BOLT_DURATION,
    BOLT_MIN_SECS,
    BOLT_SPEED_MULT,
    CATCH_DECAY_RATIO,
    CATCH_HOLD_SECS,
    CATCH_RADIUS,
    CELL_TRANSITION_SECS,
    DOOR_COMMIT_DIST,
    DOOR_GUARD_PENALTY,
    DOOR_GUARD_RADIUS,
    DOOR_ROUTE_PENALTY,
    DOOR_ROUTE_RADIUS,
    DOOR_SWITCH_HYSTERESIS,
    ESCAPE_SPEED_MULT,
    NO_LASER,
    THIEF_ANIM_FPS,
    THIEF_H,
    THIEF_W,
    TREASURE_H,
    TREASURE_W,
)
from .maze import Maze
from .pathfinding import _manhattan, find_path, get_next_step

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
        self.escape_goal: tuple[int, int] | None = None  # Door currently headed for
        self.escape_doors: list[tuple[int, int]] = []    # All candidate escape doors
        self._shared_prefix: set[tuple[int, int]] = set()  # Trunk shared by both doors
        self._escape_locked: bool = False                # Door choice committed?
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
        self._bolt_timer: float = 0.0  # >0 while panicking/sprinting from the beam

        # Walk animation (sprite sheet frames assigned by the game; may be empty).
        self.walk_frames: list[pygame.Surface] = []
        self._anim_t: float = 0.0
        self._anim_index: int = 0
        self._facing_left: bool = False

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

        # Distance from the beam to the thief this frame (None if no beam). Computed
        # once and reused for both the bolt reaction and the catch progress below.
        beam_dist = None
        if laser_active and laser_pos is not None:
            beam_dist = math.hypot(self.pos[0] - laser_pos[0], self.pos[1] - laser_pos[1])
        lit = beam_dist is not None and beam_dist < CATCH_RADIUS

        # Bolt: while the beam is on it, the thief panics and sprints to shake your
        # aim; the timer lets the sprint linger briefly after the beam slips off.
        if lit:
            self._bolt_timer = BOLT_DURATION
        else:
            self._bolt_timer = max(0.0, self._bolt_timer - dt)

        # Determine current speed based on phase, then apply the bolt sprint (floored
        # so it stays catchable even stacked on the escape speed-up).
        current_transition_secs = CELL_TRANSITION_SECS * ESCAPE_SPEED_MULT if self.phase == Phase.ESCAPE else CELL_TRANSITION_SECS
        if self._bolt_timer > 0.0:
            current_transition_secs = max(BOLT_MIN_SECS, current_transition_secs * BOLT_SPEED_MULT)

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

        # 1b) walk-cycle animation — advance frames only while actually moving.
        if self.walk_frames:
            if self.cell != self.next_cell:
                dc = self.next_cell[1] - self.cell[1]
                if dc < 0:
                    self._facing_left = True
                elif dc > 0:
                    self._facing_left = False
                self._anim_t += dt
                # Legs churn faster while bolting, so the panic sprint reads clearly.
                anim_fps = THIEF_ANIM_FPS * (1.7 if self._bolt_timer > 0.0 else 1.0)
                frame_secs = 1.0 / max(1.0, anim_fps)
                while self._anim_t >= frame_secs:
                    self._anim_t -= frame_secs
                    self._anim_index = (self._anim_index + 1) % len(self.walk_frames)
            else:
                self._anim_index = min(1, len(self.walk_frames) - 1)  # neutral pose
                self._anim_t = 0.0

        # 2) catch-progress (uses the beam distance computed at the top of update)
        if lit:
            self.catch_progress += dt / CATCH_HOLD_SECS
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
        """Set a single fixed door (used for the spawn fallback)."""
        self.escape_goal = door_pos
        self.escape_doors = [door_pos]
        self._escape_locked = True

    def set_escape_doors(self, doors: list[tuple[int, int]]) -> None:
        """Give the thief the candidate escape doors to choose between.

        The thief heads for whichever is currently safest while it is still on the
        trunk shared by both routes, then commits once the routes fork (see
        `_choose_escape_door`). Call this once, at the moment of capture.
        """
        self.escape_doors = list(doors)
        self._escape_locked = False
        # The shared trunk = cells common to the start of every door's clean path.
        paths = [find_path(self.cell, d, self.maze.is_wall, self.maze.in_bounds)
                 for d in self.escape_doors]
        paths = [p for p in paths if p]
        self._shared_prefix = self._common_prefix(paths)
        # Initial pick (no laser yet): the nearest door.
        self._choose_escape_door(None)

    @staticmethod
    def _common_prefix(paths: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
        """Cells shared by the start of every path (the route trunk)."""
        if not paths:
            return set()
        prefix: set[tuple[int, int]] = set()
        for cells in zip(*paths):
            if all(c == cells[0] for c in cells):
                prefix.add(cells[0])
            else:
                break
        return prefix

    def _steps_to(self, goal: tuple[int, int]) -> int:
        """Walk-distance (cells) from the thief's current cell to `goal`."""
        path = find_path(self.cell, goal, self.maze.is_wall, self.maze.in_bounds)
        return (len(path) - 1) if path else 10 ** 6

    def _choose_escape_door(self, laser_cell: tuple[int, int] | None) -> None:
        """Pick the safest reachable door and store it in `self.escape_goal`.

        Score = route length + a penalty if the laser guards the door or blocks the
        route ahead. Lower is safer. Hysteresis keeps the thief from flip-flopping:
        a rival door must beat the current one by more than DOOR_SWITCH_HYSTERESIS.
        """
        reachable = [d for d in self.escape_doors if self.can_reach(d)]
        if not reachable:
            self.escape_goal = self.spawn_pos  # safety: spawn is always reachable
            return

        scores: dict[tuple[int, int], int] = {}
        for d in reachable:
            path = find_path(self.cell, d, self.maze.is_wall, self.maze.in_bounds)
            if not path:
                continue
            score = len(path)
            if laser_cell is not None:
                if _manhattan(laser_cell, d) <= DOOR_GUARD_RADIUS:
                    score += DOOR_GUARD_PENALTY
                nearest_on_route = min(_manhattan(laser_cell, c) for c in path)
                if nearest_on_route <= DOOR_ROUTE_RADIUS:
                    score += DOOR_ROUTE_PENALTY * (DOOR_ROUTE_RADIUS - nearest_on_route + 1)
            scores[d] = score

        if not scores:
            self.escape_goal = self.spawn_pos
            return

        # Deterministic challenger: lowest score, ties broken by cell order.
        challenger = min(scores, key=lambda d: (scores[d], d))
        current = self.escape_goal if self.escape_goal in scores else None
        if current is None:
            self.escape_goal = challenger
        elif challenger != current and scores[challenger] < scores[current] - DOOR_SWITCH_HYSTERESIS:
            self.escape_goal = challenger
        # otherwise keep the current door (hysteresis)

    def can_reach(self, target: tuple[int, int]) -> bool:
        """True if a walkable path exists from the thief's current cell to target.
        Used to skip escape doors that are walled off (which would otherwise leave
        the thief frozen with no path to its goal)."""
        return len(find_path(self.cell, target, self.maze.is_wall, self.maze.in_bounds)) >= 1

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

        # During escape, decide which door to head for (until the choice locks).
        if self.phase == Phase.ESCAPE and self.escape_doors and not self._escape_locked:
            self._choose_escape_door(laser_cell)
            # Lock once we leave the shared trunk (past the fork) or close in on a
            # door — this kills any endgame flip-flop and gives a real catch window.
            if (self.cell not in self._shared_prefix
                    or self._steps_to(self.escape_goal) <= DOOR_COMMIT_DIST):
                self._escape_locked = True

        # Replan path
        goal = self._get_goal()
        if self.cell == goal:
            return None

        # On the final approach to the door, drop laser-avoidance: a laser parked on
        # the door cell must not make the thief refuse the last step (never escaping).
        near_goal = self.phase == Phase.ESCAPE and self._steps_to(goal) <= DOOR_COMMIT_DIST
        avoid = None if near_goal else laser_cell

        # Try with laser avoidance first
        next_step = get_next_step(
            self.cell,
            goal,
            self.maze.is_wall,
            self.maze.in_bounds,
            avoid_pos=avoid,
        )

        # Fallback: try without laser avoidance if stuck
        if next_step is None and avoid is not None:
            next_step = get_next_step(
                self.cell,
                goal,
                self.maze.is_wall,
                self.maze.in_bounds,
                avoid_pos=None,
            )

        # Safety net: if the goal is genuinely unreachable (e.g. an escape door
        # walled off from the thief), retarget to spawn so it never freezes.
        # Spawn is always reachable from the treasure where the escape begins.
        if next_step is None and self.phase == Phase.ESCAPE and goal != self.spawn_pos:
            self.escape_goal = self.spawn_pos
            self._escape_locked = True
            next_step = get_next_step(
                self.cell,
                self.spawn_pos,
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
        self.escape_doors = []
        self._shared_prefix = set()
        self._escape_locked = False
        self._bolt_timer = 0.0
        self._anim_t = 0.0
        self._anim_index = 0
        self._facing_left = False
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

        Uses the walking sprite sheet (frames assigned by game.py) when available,
        animated by `_anim_index` and mirrored by `_facing_left`. Falls back to a
        plain rectangle if no art is loaded.
        """
        if self.walk_frames:
            frame = self.walk_frames[self._anim_index]
            if self._facing_left:
                frame = pygame.transform.flip(frame, True, False)
            rect = frame.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
            surface.blit(frame, rect.topleft)
            return

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
