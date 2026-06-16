"""
Template game — the smallest complete Scene. Copy this folder to start a new game.

It shows the three things every game does:
  1. read the Pointer each frame (the only player input),
  2. update game state from it,
  3. draw to the canvas.

Run it with:  python run.py --game _template --windowed
Full walkthrough: docs/BUILDING_A_NEW_GAME.md
"""

from __future__ import annotations

from pathlib import Path

import pygame

from arcade.config import GAME_H, GAME_W
from arcade.scene import Scene
from arcade import gestures, palette
from arcade.assets import AssetLoader


class TemplateGame(Scene):
    # The engine draws a default pointer dot when this is True. Set False if your
    # game shows the pointer its own way (the thief game uses a spotlight).
    show_pointer = False

    def __init__(self, app):
        super().__init__(app)
        # Load art/sound from THIS folder's assets/. Missing files become
        # labeled placeholders, so the game runs before any art exists.
        self.assets = AssetLoader(Path(__file__).with_name("assets"))
        self.font = pygame.font.SysFont("monospace", 40, bold=True)

        self.trail = gestures.Trail(max_points=14)
        self.marker = (GAME_W / 2, GAME_H / 2)
        self.lit = False

    # Operator/dev keys only — never gameplay (the player has no keyboard).
    def handle_event(self, event: pygame.event.Event) -> None:
        pass

    def update(self, dt: float, pointer) -> None:
        self.lit = pointer.active
        if pointer.active:
            self.marker = pointer.pos
        self.trail.update(pointer)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(palette.BLACK)

        title = self.font.render("template game — move the pointer", True, palette.SLATE)
        surface.blit(title, (GAME_W // 2 - title.get_width() // 2, 80))

        # Draw the swipe trail (a non-green color, of course).
        self.trail.draw(surface, palette.CYAN, width=10)

        # Draw the pointer marker. Replace this with your own sprite, e.g.:
        #   img = self.assets.load_image("blade.png", size=(120, 120))
        #   surface.blit(img, (self.marker[0] - 60, self.marker[1] - 60))
        color = palette.GOLD if self.lit else palette.SLATE
        pygame.draw.circle(surface, color, (int(self.marker[0]), int(self.marker[1])), 28, 4)


def create(app):
    """Factory used by run.py's game registry."""
    return TemplateGame(app)
