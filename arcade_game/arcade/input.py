"""
The ONE input every game is allowed to use.

A game receives a `Pointer` each frame and nothing else for player control. The
Pointer is everything a real laser pointer can tell us:

    - where it is on the screen   (x, y, in 1920x1080 canvas space)
    - whether it is on the screen at all this frame   (active)
    - how fast it is moving        (vx, vy, pixels since last frame)

That is the whole contract. There is deliberately no mouse-button, no keyboard,
no scroll wheel, and no second cursor here — because a laser pointer has none of
those. If a mechanic can't be expressed with a moving dot that can blink out,
it cannot run on the real installation. See docs/INPUT_CONSTRAINTS.md.

On a dev machine the pointer is the mouse (`MousePointer`). On the real system a
`LaserPointer` (in the separate detection project) implements the SAME interface,
so games port across with zero change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pygame

from arcade.config import GAME_H, GAME_W


@dataclass
class Pointer:
    """A snapshot of the pointer for a single frame (canvas coordinates)."""

    x: float
    y: float
    active: bool          # is the pointer on the screen this frame?
    vx: float = 0.0       # horizontal movement since the previous frame (px)
    vy: float = 0.0       # vertical movement since the previous frame (px)

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def speed(self) -> float:
        """Distance moved since the previous frame, in pixels."""
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5


class PointerSource(ABC):
    """Anything that can produce a Pointer once per frame."""

    @abstractmethod
    def poll(self) -> Pointer:
        ...


class MousePointer(PointerSource):
    """
    Mouse stand-in for the laser, for development.

    - position  -> the mouse position mapped into canvas (1920x1080) space
    - active    -> True while the mouse is inside the focused window;
                   moving the mouse out of the window emulates the laser
                   leaving the projected surface (the pointer goes dark).
    - hold_to_emit -> if True, the pointer is only active while the left mouse
                   button is held. Useful to rehearse the laser blinking on and
                   off; off by default so normal play feels natural.
    """

    def __init__(self, display, hold_to_emit: bool = False):
        self._display = display
        self._hold_to_emit = hold_to_emit
        self._prev: tuple[float, float] | None = None

    def poll(self) -> Pointer:
        mx, my = pygame.mouse.get_pos()
        cx, cy = self._display.to_canvas(mx, my)

        in_bounds = (0.0 <= cx <= GAME_W) and (0.0 <= cy <= GAME_H)
        active = bool(pygame.mouse.get_focused()) and in_bounds
        if self._hold_to_emit:
            active = active and pygame.mouse.get_pressed()[0]

        vx = vy = 0.0
        if self._prev is not None:
            vx = cx - self._prev[0]
            vy = cy - self._prev[1]
        self._prev = (cx, cy)

        # Clamp reported position to the canvas so games never see off-screen
        # coordinates (the laser can only ever be reported on the surface).
        cx = max(0.0, min(float(GAME_W), cx))
        cy = max(0.0, min(float(GAME_H), cy))
        return Pointer(cx, cy, active, vx, vy)
