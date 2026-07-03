"""
Window + virtual canvas.

Every game draws onto a fixed 1920x1080 `canvas` surface. The Display scales
that canvas to whatever the real window is (a small dev window, or the full
projector) with letterboxing, and maps mouse coordinates back into canvas space
so the pointer is always correct. Games therefore use one coordinate system and
never worry about the window size.
"""

from __future__ import annotations

import os

import pygame

from arcade.config import GAME_H, GAME_W, WINDOW_H, WINDOW_W


def enable_dpi_awareness() -> None:
    """
    Windows: opt into DPI awareness so a 1920x1080 fullscreen window maps 1:1 to
    physical pixels. Without this, display scaling >100% makes the window render
    larger than the screen and crops the bottom/right edges. Must run before the
    window is created. (Ported from the real installation, where it fixed exactly
    that cropping on the projector laptop.)
    """
    if os.name != "nt":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class Display:
    def __init__(self, windowed: bool = False):
        enable_dpi_awareness()
        pygame.init()

        if windowed:
            self.win_w, self.win_h = WINDOW_W, WINDOW_H
            self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        else:
            self.screen = pygame.display.set_mode((GAME_W, GAME_H), pygame.FULLSCREEN)
            self.win_w, self.win_h = self.screen.get_size()

        pygame.display.set_caption("laser-arcade")

        # The fixed surface all games draw onto (created after the display mode
        # exists, so .convert() has a pixel format to match).
        self.canvas = pygame.Surface((GAME_W, GAME_H)).convert()
        self._recompute()

    def _recompute(self) -> None:
        # Largest centered rectangle of canvas aspect that fits the window.
        self.scale = min(self.win_w / GAME_W, self.win_h / GAME_H)
        self.draw_w = int(GAME_W * self.scale)
        self.draw_h = int(GAME_H * self.scale)
        self.off_x = (self.win_w - self.draw_w) // 2
        self.off_y = (self.win_h - self.draw_h) // 2

        if self.scale == 1 and self.off_x == 0 and self.off_y == 0:
            # Window matches the canvas exactly — present() direct-blits.
            self._scaled_buf = None
        else:
            # Persistent destination for the per-frame scale. Allocating this
            # every frame (as smoothscale used to) was a big main-thread cost
            # that starved the YOLO detector thread of CPU/GIL time.
            self._scaled_buf = pygame.Surface((self.draw_w, self.draw_h)).convert()
            # The black letterbox bars never change between resizes, so fill
            # them once here instead of screen.fill() every frame.
            self.screen.fill((0, 0, 0))

    def handle_resize(self, w: int, h: int) -> None:
        self.win_w, self.win_h = max(320, w), max(180, h)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        self._recompute()

    def to_canvas(self, win_x: float, win_y: float) -> tuple[float, float]:
        """Map a window/mouse coordinate into canvas (1920x1080) space."""
        if self.scale <= 0:
            return (0.0, 0.0)
        return ((win_x - self.off_x) / self.scale,
                (win_y - self.off_y) / self.scale)

    def present(self) -> None:
        """Blit the canvas to the window (scaled + letterboxed) and flip."""
        if self._scaled_buf is None:
            self.screen.blit(self.canvas, (0, 0))
        else:
            # Nearest-neighbour scale into a reused buffer. smoothscale here
            # allocated a fresh window-sized surface and did a bicubic filter
            # of the whole 1080p canvas EVERY frame — the single biggest
            # main-thread cost, and it starved the detector thread (see
            # DETECTION_POSTMORTEM.md in the deployment repo). The quality
            # difference is invisible in motion on a projector.
            pygame.transform.scale(self.canvas, (self.draw_w, self.draw_h), self._scaled_buf)
            self.screen.blit(self._scaled_buf, (self.off_x, self.off_y))
        pygame.display.flip()
