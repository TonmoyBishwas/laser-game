"""
Scene base class, the main loop (GameApp), and the game-picker menu.

A game is a Scene: it gets `update(dt, pointer)` and `draw(canvas)` each frame.
GameApp owns the window and the pointer, runs the loop, and switches between
scenes (menu <-> games). Keyboard here is for the OPERATOR only — quitting,
restarting, returning to the menu — never for gameplay.
"""

from __future__ import annotations

import pygame

from arcade.config import TARGET_FPS
from arcade.display import Display
from arcade.dwell import DwellTracker
from arcade.input import MousePointer, Pointer
from arcade import palette


class Scene:
    """Base class for every game/screen. Subclass and override the hooks."""

    # If True, GameApp draws a small pointer dot on top of the scene. Games that
    # show the pointer their own way (e.g. a spotlight) set this False.
    show_pointer: bool = True

    def __init__(self, app: "GameApp"):
        self.app = app

    def handle_event(self, event: pygame.event.Event) -> None:
        """Operator/dev key events only (e.g. R to restart). Not for gameplay."""

    def update(self, dt: float, pointer: Pointer) -> None:
        ...

    def draw(self, surface: pygame.Surface) -> None:
        ...


class GameApp:
    """Owns the window + pointer and drives the active Scene."""

    def __init__(self, registry: dict, windowed: bool = False, hold_to_emit: bool = False):
        self.registry = registry                       # name -> factory(app) -> Scene
        self.display = Display(windowed=windowed)
        self.pointer_source = MousePointer(self.display, hold_to_emit=hold_to_emit)
        self.clock = pygame.time.Clock()
        self.scene: Scene | None = None
        self._running = False

    # ── Scene control ───────────────────────────────────────────────────────
    def launch(self, name: str) -> None:
        factory = self.registry[name]
        self.scene = factory(self)

    def menu(self) -> None:
        self.scene = MenuScene(self)

    def quit(self) -> None:
        self._running = False

    # ── Main loop ───────────────────────────────────────────────────────────
    def run(self, start: Scene) -> None:
        self.scene = start
        self._running = True
        while self._running:
            dt = self.clock.tick(TARGET_FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.display.handle_resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self._running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    self.menu()
                else:
                    if self.scene is not None:
                        self.scene.handle_event(event)

            pointer = self.pointer_source.poll()
            if self.scene is not None:
                self.scene.update(dt, pointer)
                self.scene.draw(self.display.canvas)
                if self.scene.show_pointer and pointer.active:
                    pygame.draw.circle(self.display.canvas, palette.WHITE,
                                       (int(pointer.x), int(pointer.y)), 10, 2)

            self.display.present()

        pygame.quit()


class MenuScene(Scene):
    """Pick a game by resting the pointer on its tile (dwell to select)."""

    show_pointer = True

    def __init__(self, app: GameApp):
        super().__init__(app)
        # Offer every registered game except the copy-me template.
        self.names = [n for n in app.registry if not n.startswith("_")]
        self.font_title = pygame.font.SysFont("monospace", 64, bold=True)
        self.font_item = pygame.font.SysFont("monospace", 40, bold=True)
        self.font_hint = pygame.font.SysFont("monospace", 24)
        self.dwell = DwellTracker(hold_secs=0.8, tolerance=160.0)
        self._hover: str | None = None
        self._tiles: dict[str, pygame.Rect] = {}

        from arcade.config import GAME_H, GAME_W
        n = max(1, len(self.names))
        tile_w, tile_h, gap = 520, 130, 30
        total_h = n * tile_h + (n - 1) * gap
        top = GAME_H // 2 - total_h // 2 + 40
        for i, name in enumerate(self.names):
            rect = pygame.Rect(0, 0, tile_w, tile_h)
            rect.center = (GAME_W // 2, top + i * (tile_h + gap) + tile_h // 2)
            self._tiles[name] = rect

        # All the menu text is static — render it once instead of every frame
        # (repeated font.render calls are needless main-thread work).
        self._title_surf = self.font_title.render("laser-arcade", True, palette.CYAN)
        self._hint_surf = self.font_hint.render(
            "Rest the pointer on a game to start  |  M menu  |  Q quit",
            True, palette.SLATE)
        self._labels = {
            name: self.font_item.render(name.replace("_", " ").title(), True, palette.WHITE)
            for name in self.names
        }

    def update(self, dt: float, pointer: Pointer) -> None:
        # Which tile is the pointer over?
        self._hover = None
        for name, rect in self._tiles.items():
            if rect.collidepoint(pointer.pos):
                self._hover = name
                break

        if self._hover is None:
            self.dwell.progress = max(0.0, self.dwell.progress - dt)
            return

        target = self._tiles[self._hover].center
        if self.dwell.update(dt, pointer, target):
            self.app.launch(self._hover)

    def draw(self, surface: pygame.Surface) -> None:
        from arcade.config import GAME_H, GAME_W
        surface.fill(palette.BLACK)

        title = self._title_surf
        surface.blit(title, (GAME_W // 2 - title.get_width() // 2, 150))

        for name, rect in self._tiles.items():
            hovered = (name == self._hover)
            border = palette.GOLD if hovered else palette.SLATE
            pygame.draw.rect(surface, (20, 24, 38), rect, border_radius=16)
            pygame.draw.rect(surface, border, rect, width=4, border_radius=16)

            label = self._labels[name]
            surface.blit(label, (rect.centerx - label.get_width() // 2,
                                 rect.centery - label.get_height() // 2))

            if hovered and self.dwell.progress > 0:
                bar_w = int((rect.width - 40) * self.dwell.progress)
                pygame.draw.rect(surface, palette.GOLD,
                                 (rect.left + 20, rect.bottom - 16, bar_w, 8), border_radius=4)

        hint = self._hint_surf
        surface.blit(hint, (GAME_W // 2 - hint.get_width() // 2, GAME_H - 80))
