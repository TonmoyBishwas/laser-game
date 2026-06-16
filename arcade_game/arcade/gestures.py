"""
Motion helpers — the pointer's vocabulary beyond just "where is it".

A laser pointer can't click, but it CAN move fast through things. That single
fact powers slicing (Fruit-Ninja-style), sweeping, and flick gestures. These
helpers work purely from pointer positions and velocity, so anything built on
them runs identically with the mouse and the real laser.
"""

from __future__ import annotations

import pygame

from arcade.input import Pointer


def segment_hits_circle(
    p0: tuple[float, float],
    p1: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    """
    Did the straight path from p0 to p1 pass within `radius` of `center`?

    This is THE slice test: p0 is where the pointer was last frame, p1 is where
    it is now, and `center`/`radius` describe a target (a fruit). It returns True
    if the swipe crossed the target — no click required. Checking the whole
    segment (not just the current point) means fast swipes never "tunnel" past a
    target between frames.
    """
    cx, cy = center
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq <= 1e-9:
        # No movement: fall back to a point-in-circle test.
        ddx, ddy = x0 - cx, y0 - cy
        return (ddx * ddx + ddy * ddy) <= radius * radius

    # Project center onto the segment, clamped to [0, 1].
    t = ((cx - x0) * dx + (cy - y0) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    nearest_x = x0 + t * dx
    nearest_y = y0 + t * dy
    ddx, ddy = nearest_x - cx, nearest_y - cy
    return (ddx * ddx + ddy * ddy) <= radius * radius


def is_fast(pointer: Pointer, min_speed: float = 25.0) -> bool:
    """
    True if the pointer is moving fast enough to count as an intentional swipe
    (pixels per frame). Use this to gate slicing so a slow, resting pointer
    doesn't accidentally clear the screen.
    """
    return pointer.active and pointer.speed >= min_speed


class Trail:
    """
    Keeps the last N pointer positions so you can draw a blade/comet trail and
    run slice tests across the most recent movement. Purely visual + helper data;
    no input state of its own beyond history.
    """

    def __init__(self, max_points: int = 12):
        self.max_points = max_points
        self.points: list[tuple[float, float]] = []

    def update(self, pointer: Pointer) -> None:
        if pointer.active:
            self.points.append(pointer.pos)
            if len(self.points) > self.max_points:
                self.points.pop(0)
        else:
            # Pointer went dark — break the trail so it doesn't draw a long jump.
            self.points.clear()

    def last_segment(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """The most recent (previous, current) movement segment, or None."""
        if len(self.points) < 2:
            return None
        return self.points[-2], self.points[-1]

    def draw(self, surface: pygame.Surface, color, width: int = 6) -> None:
        """Draw the trail as a fading polyline. Use a non-green color."""
        if len(self.points) < 2:
            return
        n = len(self.points)
        for i in range(1, n):
            a = int(255 * i / n)
            seg_w = max(1, int(width * i / n))
            p0, p1 = self.points[i - 1], self.points[i]
            pygame.draw.line(surface, color, p0, p1, seg_w)
            _ = a  # alpha-by-segment can be added with a per-segment surface
