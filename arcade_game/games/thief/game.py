"""
Thief game — maze + flashlight reveal + pathfinding-driven thief, as a Scene.

This is the same game as the real installation, with one change: instead of a
laser position it reads the engine `Pointer` (mouse in dev, laser for real). The
pointer being `active` replaces the laser "freshness" check — everything else is
the original logic.

Two-phase gameplay:
1. Thief navigates from spawn to treasure (avoiding player's spotlight)
2. After treasure, thief escapes to its spawn door (faster, door appears)

Render order each frame:
  1. maze surface (walls)        5. treasure (always visible)
  2. thief body                  6. catch ring
  3. dark overlay punched by spotlight  7. door (if visible)
  4.                             8. edge glow  9. HUD / end screen
"""

import math
from pathlib import Path
from typing import Any

import pygame

from arcade.assets import AssetLoader
from arcade.scene import Scene

from .constants import (
    CELL_H,
    CELL_W,
    DOOR_H,
    DOOR_W,
    EDGE_GLOW_BASE,
    EDGE_GLOW_DEPTH,
    EDGE_GLOW_PULSE,
    GAME_H,
    GAME_W,
    GRID_COLS,
    GRID_ROWS,
    ROUND_TIMEOUT_SECS,
    SPOTLIGHT_FADE_SECS,
    SPOTLIGHT_RADIUS,
    TREASURE_H,
    TREASURE_W,
)
from .maze import Maze
from .maze_layouts import DEFAULT, random_layout
from .thief import Phase, Thief

# ── Colours (all green-free; green belongs to the laser) ────────────────────────
BG_COLOR          = (8, 8, 18)
WALL_COLOR        = (60, 160, 220)
WALL_INNER_COLOR  = (15, 60, 100)
TREASURE_COLOR    = (255, 215, 0)
TREASURE_GLOW     = (255, 240, 100)
HUD_COLOR         = (220, 220, 220)
OVERLAY_ALPHA_MAX = 245
EDGE_GLOW_COLOR   = (220, 50, 50)


# ── Spotlight + overlay caches (reused every frame) ─────────────────────────────
def _build_spotlight_surf(radius: int) -> pygame.Surface:
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        a = int(255 * (1 - r / radius) ** 0.6)
        pygame.draw.circle(surf, (255, 255, 200, a), (radius, radius), r)
    return surf


class ThiefGame(Scene):
    show_pointer = False   # the spotlight is the cursor

    def __init__(self, app):
        super().__init__(app)
        self._font_large = pygame.font.SysFont("monospace", 72, bold=True)
        self._font_med   = pygame.font.SysFont("monospace", 36, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 22)

        # Asset loader for sprites
        self._assets = AssetLoader(Path(__file__).with_name("assets"))

        # Load custom assets
        self.background_img: pygame.Surface | None = None
        self.treasure_img: pygame.Surface | None = None
        self.treasure_stolen_img: pygame.Surface | None = None
        self._load_assets()

        # Game state
        self.maze: Maze | None = None
        self.maze_surface: pygame.Surface | None = None
        self.thief: Thief | None = None
        self.spawn_pos: tuple[int, int] | None = None
        self.door_positions: list[tuple[int, int]] = []  # All escape doors
        self.active_escape_door: tuple[int, int] | None = None  # The door thief will use
        self.door_visible: bool = False
        self.door_img: pygame.Surface | None = None
        self.use_random_mazes: bool = True  # Set to False to use DEFAULT layout
        self.num_doors: int = 4  # Number of escape doors to generate

        self._spotlight = _build_spotlight_surf(SPOTLIGHT_RADIUS)
        self._overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)

        self.elapsed = 0.0
        self.game_over = False
        self.player_won = False
        self.spot_alpha = 0.0
        self.spot_pos = (GAME_W / 2, GAME_H / 2)
        self._fade_rate = 255.0 / max(SPOTLIGHT_FADE_SECS, 0.01)

        # Treasure captured visual feedback
        self.treasure_captured_time: float | None = None
        self.capture_flash_alpha: float = 0.0
        self.screen_shake: float = 0.0

        # Initialize the game
        self._restart()

    def _load_assets(self) -> None:
        """Load custom assets (background, treasure sprites)."""
        try:
            self.background_img = self._assets.load_image("background.png", size=(GAME_W, GAME_H))
        except Exception:
            self.background_img = None  # Fallback to solid color

        try:
            self.treasure_img = self._assets.load_image("treasure.png", size=(TREASURE_W, TREASURE_H))
        except Exception:
            self.treasure_img = None  # Fallback to drawn treasure

        try:
            self.treasure_stolen_img = self._assets.load_image("treasure_stolen.png", size=(TREASURE_W, TREASURE_H))
        except Exception:
            self.treasure_stolen_img = None

    # ── Setup helpers ───────────────────────────────────────────────────────
    def _generate_maze(self) -> list[list[Any]]:
        """Generate a maze layout (random or default)."""
        if self.use_random_mazes:
            try:
                return random_layout()
            except Exception as e:
                print(f"[thief] Random maze generation failed: {e}, using DEFAULT")
                return DEFAULT
        return DEFAULT

    def _pick_positions(self, grid: list[list[Any]]) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
        """
        Pick spawn, treasure, and door positions from a grid.
        Returns (spawn, treasure, doors) as tuples of (row, col).
        For DEFAULT layout, reads the 'S' and 'T' markers.
        For random grids, picks random open cells.
        """
        # Check if grid has explicit S/T/D markers
        spawn = None
        treasure = None
        doors = []
        open_cells = []

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                val = grid[r][c]
                if val == 'S':
                    spawn = (r, c)
                elif val == 'T':
                    treasure = (r, c)
                elif val == 'D':
                    doors.append((r, c))
                elif val == 0:
                    open_cells.append((r, c))

        # If no explicit markers, pick random positions
        if spawn is None and open_cells:
            import random
            if len(open_cells) >= 2:
                spawn, treasure = random.sample(open_cells, 2)
            else:
                spawn, treasure = open_cells[0], open_cells[0]

        # Fallback to defaults if still None
        if spawn is None:
            spawn = (GRID_ROWS - 2, GRID_COLS // 2)
        if treasure is None:
            treasure = (1, GRID_COLS // 2)

        # If no doors in grid, add random doors for DEFAULT layout
        if not doors and grid != DEFAULT:
            available_for_doors = [(r, c) for r in range(1, GRID_ROWS - 1)
                                   for c in range(1, GRID_COLS - 1)
                                   if grid[r][c] == OPEN and (r, c) != spawn and (r, c) != treasure]
            doors_to_add = min(self.num_doors, len(available_for_doors))
            if doors_to_add > 0:
                doors = random.sample(available_for_doors, doors_to_add)

        return spawn, treasure, doors

    def _build_maze_surface(self) -> pygame.Surface:
        surf = pygame.Surface((GAME_W, GAME_H))
        # Use background image if available, otherwise fallback to solid color
        if self.background_img is not None:
            surf.blit(self.background_img, (0, 0))
        else:
            surf.fill(BG_COLOR)
        pad = 4
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.maze.is_wall(r, c):
                    rect = pygame.Rect(c * CELL_W + pad, r * CELL_H + pad,
                                       CELL_W - 2 * pad, CELL_H - 2 * pad)
                    pygame.draw.rect(surf, WALL_INNER_COLOR, rect)
                    pygame.draw.rect(surf, WALL_COLOR, rect, width=3)
        return surf

    # ── Input (operator only) ─────────────────────────────────────────────────
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and self.game_over:
                self._restart()
            elif event.key == pygame.K_m:
                # Toggle between random and fixed mazes for testing
                self.use_random_mazes = not self.use_random_mazes
                print(f"[thief] Random mazes: {self.use_random_mazes}")

    def _restart(self) -> None:
        """Start a new round with fresh maze and positions."""
        # Generate maze
        layout = self._generate_maze()
        spawn, treasure, doors = self._pick_positions(layout)

        # Update layout with positions if using random generation
        if layout != DEFAULT:
            layout[spawn[0]][spawn[1]] = 'S'
            layout[treasure[0]][treasure[1]] = 'T'

        self.maze = Maze(layout)
        self.spawn_pos = spawn
        self.door_positions = doors
        self.active_escape_door = None
        self.maze_surface = self._build_maze_surface()

        # Create thief with spawn position
        self.thief = Thief(self.maze, spawn_pos=spawn)

        # Load door image (cached, will show placeholder if missing)
        self.door_img = self._assets.load_image("door.png", size=(DOOR_W, DOOR_H))

        # Reset game state
        self.elapsed = 0.0
        self.game_over = False
        self.player_won = False
        self.door_visible = False
        self.spot_alpha = 0.0
        self.treasure_captured_time = None
        self.capture_flash_alpha = 0.0
        self.screen_shake = 0.0

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self, dt: float, pointer) -> None:
        if self.game_over:
            return

        # Pointer drives the flashlight. `active` replaces the laser freshness check.
        if pointer.active:
            self.spot_pos = pointer.pos
            self.spot_alpha = min(255.0, self.spot_alpha + self._fade_rate * dt * 4)
        else:
            self.spot_alpha = max(0.0, self.spot_alpha - self._fade_rate * dt)
        laser_active = self.spot_alpha > 128

        laser_cell = None
        if pointer.active:
            laser_cell = self.maze.pixel_to_cell(self.spot_pos[0], self.spot_pos[1])

        self.elapsed += dt
        self.thief.update(dt, self.spot_pos if laser_active else None, laser_active, laser_cell)

        # Check for treasure reached - switch to escape phase
        if self.thief.reached_treasure and not self.door_visible:
            self.door_visible = True
            self.treasure_captured_time = self.elapsed  # Track capture time for effects
            self.capture_flash_alpha = 255.0  # Start flash effect
            self.screen_shake = 10.0  # Screen shake magnitude

            # Pick a random door for the thief to escape through
            if self.door_positions:
                import random
                self.active_escape_door = random.choice(self.door_positions)
                # Set thief's escape goal to this door
                self.thief.set_escape_goal(self.active_escape_door)
            else:
                # Fallback to spawn position if no doors
                self.active_escape_door = self.spawn_pos

        # Update visual effects
        if self.capture_flash_alpha > 0:
            self.capture_flash_alpha -= dt * 150  # Fade out flash
            if self.capture_flash_alpha < 0:
                self.capture_flash_alpha = 0

        if self.screen_shake > 0:
            self.screen_shake -= dt * 20  # Reduce shake
            if self.screen_shake < 0:
                self.screen_shake = 0

        # End conditions
        if self.thief.caught:
            self.game_over, self.player_won = True, True
        elif self.thief.escaped:
            self.game_over, self.player_won = True, False
        elif self.elapsed >= ROUND_TIMEOUT_SECS:
            self.game_over, self.player_won = True, False

    # ── Draw ────────────────────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface) -> None:
        # Apply screen shake if active
        shake_x = 0
        shake_y = 0
        if self.screen_shake > 0:
            import random
            shake_x = random.uniform(-self.screen_shake, self.screen_shake)
            shake_y = random.uniform(-self.screen_shake, self.screen_shake)

        # Create a temporary surface for the shaken view
        if self.screen_shake > 0:
            temp_surface = pygame.Surface((GAME_W, GAME_H))
            temp_surface.blit(self.maze_surface, (0, 0))
            self.thief.draw(temp_surface)
            # Draw stolen treasure on thief during escape
            if self.door_visible:
                self.thief.draw_stolen_treasure(temp_surface, self.treasure_stolen_img)
            self._draw_overlay(temp_surface)
            self._draw_treasure(temp_surface)
            self._draw_door(temp_surface)
            self.thief.draw_catch_ring(temp_surface)
            self._draw_edge_glow(temp_surface)
            self._draw_hud(temp_surface)
            surface.blit(temp_surface, (shake_x, shake_y))
        else:
            surface.blit(self.maze_surface, (0, 0))
            self.thief.draw(surface)
            # Draw stolen treasure on thief during escape
            if self.door_visible:
                self.thief.draw_stolen_treasure(surface, self.treasure_stolen_img)
            self._draw_overlay(surface)
            self._draw_treasure(surface)
            self._draw_door(surface)
            self.thief.draw_catch_ring(surface)
            self._draw_edge_glow(surface)
            self._draw_hud(surface)

        # Draw treasure capture flash effect
        if self.capture_flash_alpha > 0:
            flash_surface = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            flash_alpha = int(self.capture_flash_alpha)
            flash_surface.fill((255, 100, 50, flash_alpha))
            surface.blit(flash_surface, (0, 0))

        # Draw "ESCAPE!" announcement
        if self.door_visible and not self.game_over:
            time_since_capture = self.elapsed - (self.treasure_captured_time or 0)
            if time_since_capture < 3.0:  # Show for 3 seconds
                # Pulsing "ESCAPE!" text
                pulse = 1.0 + 0.2 * abs(math.sin(self.elapsed * 5))
                font_size = int(72 * pulse)
                escape_font = pygame.font.SysFont("monospace", font_size, bold=True)
                text = escape_font.render("ESCAPE!", True, (255, 80, 80))
                text_rect = text.get_rect(center=(GAME_W // 2, 100))

                # Draw outline
                outline_color = (0, 0, 0)
                for dx in [-3, -2, -1, 1, 2, 3]:
                    for dy in [-3, -2, -1, 1, 2, 3]:
                        outline = escape_font.render("ESCAPE!", True, outline_color)
                        surface.blit(outline, (text_rect.x + dx, text_rect.y + dy))

                # Draw main text
                surface.blit(text, text_rect)

                # Draw secondary instruction
                sub_font = pygame.font.SysFont("monospace", 28, bold=True)
                sub_text = sub_font.render("Find the door! Catch the thief!", True, (255, 200, 100))
                sub_rect = sub_text.get_rect(center=(GAME_W // 2, 160))
                surface.blit(sub_text, sub_rect)

        if self.game_over:
            self._draw_end_screen(surface)

    def _draw_overlay(self, surface: pygame.Surface) -> None:
        self._overlay.fill((0, 0, 0, OVERLAY_ALPHA_MAX))
        hole_alpha = int(self.spot_alpha)
        if hole_alpha > 5:
            scaled = self._spotlight.copy()
            scaled.fill((255, 255, 255, hole_alpha), special_flags=pygame.BLEND_RGBA_MULT)
            self._overlay.blit(
                scaled,
                (int(self.spot_pos[0]) - SPOTLIGHT_RADIUS, int(self.spot_pos[1]) - SPOTLIGHT_RADIUS),
                special_flags=pygame.BLEND_RGBA_SUB,
            )
        surface.blit(self._overlay, (0, 0))

    def _draw_treasure(self, surface: pygame.Surface) -> None:
        # Treasure disappears after thief grabs it
        if self.door_visible:
            return

        cx, cy = self.maze.cell_center(*self.maze.treasure)
        rect = pygame.Rect(int(cx - TREASURE_W // 2), int(cy - TREASURE_H // 2),
                           TREASURE_W, TREASURE_H)

        if self.treasure_img is not None:
            # Draw the treasure sprite with a pulsing glow effect
            pulse = int(abs(math.sin(self.elapsed * 2)) * 10)
            glow_rect = rect.inflate(pulse * 2, pulse * 2)

            # Draw glow outline
            pygame.draw.rect(surface, TREASURE_GLOW, glow_rect,
                           width=3, border_radius=12)

            # Draw the sprite
            surface.blit(self.treasure_img, rect.topleft)

            # Optional label
            label = self._font_small.render("TREASURE", True, (40, 30, 0))
            surface.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom + 4))
        else:
            # Fallback to drawn treasure
            pygame.draw.rect(surface, TREASURE_COLOR, rect, border_radius=10)
            pulse = int(abs(math.sin(self.elapsed * 2)) * 8)
            pygame.draw.rect(surface, TREASURE_GLOW, rect.inflate(pulse * 2, pulse * 2),
                             width=3, border_radius=12)
            label = self._font_small.render("TREASURE", True, (40, 30, 0))
            surface.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom + 4))

    def _draw_door(self, surface: pygame.Surface) -> None:
        """Draw all escape doors (visible only when in spotlight)."""
        if not self.door_visible:
            return

        # Draw all doors
        doors_to_draw = self.door_positions if self.door_positions else [self.spawn_pos]

        for door_pos in doors_to_draw:
            if door_pos is None:
                continue

            cx, cy = self.maze.cell_center(*door_pos)

            # Check if door is illuminated by the spotlight
            dist_to_spotlight = math.hypot(cx - self.spot_pos[0], cy - self.spot_pos[1])
            is_illuminated = dist_to_spotlight < SPOTLIGHT_RADIUS and self.spot_alpha > 50

            if not is_illuminated:
                continue  # Door is hidden in darkness

            # Calculate glow intensity based on distance to spotlight center
            glow_intensity = 1.0 - min(1.0, dist_to_spotlight / SPOTLIGHT_RADIUS)

            # Check if this is the active escape door
            is_active_door = (door_pos == self.active_escape_door)

            if self.door_img is not None:
                # Draw the sprite with a glow effect
                rect = pygame.Rect(int(cx - DOOR_W // 2), int(cy - DOOR_H // 2), DOOR_W, DOOR_H)

                # Draw glow outline (pulsing) - stronger for active door
                pulse_strength = 10 if is_active_door else 6
                pulse = int(abs(math.sin(self.elapsed * 3)) * pulse_strength * glow_intensity)
                glow_color = (255, 150, 50) if is_active_door else (255, 200, 100)
                glow_rect = rect.inflate(pulse * 2, pulse * 2)
                pygame.draw.rect(surface, glow_color, glow_rect, width=3 if is_active_door else 2, border_radius=8)

                # Draw the sprite
                surface.blit(self.door_img, rect.topleft)

                # Add indicator for active door
                if is_active_door:
                    # Draw an arrow or indicator above the active door
                    arrow_size = 12
                    arrow_y = rect.top - arrow_size - 5
                    pygame.draw.polygon(surface, (255, 100, 50), [
                        (rect.centerx, arrow_y),
                        (rect.centerx - arrow_size // 2, arrow_y + arrow_size),
                        (rect.centerx + arrow_size // 2, arrow_y + arrow_size)
                    ])
            else:
                # Fallback: draw a placeholder door with glow
                rect = pygame.Rect(int(cx - DOOR_W // 2), int(cy - DOOR_H // 2), DOOR_W, DOOR_H)
                door_color = (160, 100, 60) if is_active_door else (139, 90, 50)
                pygame.draw.rect(surface, door_color, rect, border_radius=5)
                pygame.draw.rect(surface, (100, 60, 30), rect, width=3, border_radius=5)

                # Glow effect when illuminated
                glow_size = int(12 * glow_intensity if is_active_door else 8 * glow_intensity)
                pygame.draw.rect(surface, (255, 200, 100),
                               rect.inflate(glow_size * 2, glow_size * 2),
                               width=3 if is_active_door else 2, border_radius=8)
            glow_size = int(8 * glow_intensity)
            pygame.draw.rect(surface, (255, 200, 100),
                           rect.inflate(glow_size * 2, glow_size * 2),
                           width=2, border_radius=8)

            # Door handle
            pygame.draw.circle(surface, (200, 180, 100), (rect.right - 10, rect.centery), 4)

    def _draw_edge_glow(self, surface: pygame.Surface) -> None:
        tx, ty = self.thief.pos
        t = self.elapsed
        pulse = EDGE_GLOW_BASE + EDGE_GLOW_PULSE * abs(math.sin(t * 3.0))
        edges = {
            "top":    1.0 - min(1.0, ty / (GAME_H * 0.5)),
            "bottom": 1.0 - min(1.0, (GAME_H - ty) / (GAME_H * 0.5)),
            "left":   1.0 - min(1.0, tx / (GAME_W * 0.5)),
            "right":  1.0 - min(1.0, (GAME_W - tx) / (GAME_W * 0.5)),
        }
        depth = EDGE_GLOW_DEPTH
        for name, prox in edges.items():
            if prox <= 0.01:
                continue
            max_alpha = int(min(255, prox * pulse * 255))
            if max_alpha <= 0:
                continue
            if name in ("top", "bottom"):
                strip = pygame.Surface((GAME_W, depth), pygame.SRCALPHA)
                for i in range(depth):
                    a = int(max_alpha * (1 - i / depth))
                    row = i if name == "top" else depth - 1 - i
                    pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (0, row), (GAME_W, row))
                surface.blit(strip, (0, 0 if name == "top" else GAME_H - depth))
            else:
                strip = pygame.Surface((depth, GAME_H), pygame.SRCALPHA)
                for i in range(depth):
                    a = int(max_alpha * (1 - i / depth))
                    col = i if name == "left" else depth - 1 - i
                    pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (col, 0), (col, GAME_H))
                surface.blit(strip, (0 if name == "left" else GAME_W - depth, 0))

    def _draw_hud(self, surface: pygame.Surface) -> None:
        remaining = max(0.0, ROUND_TIMEOUT_SECS - self.elapsed)
        txt = self._font_med.render(f"Time: {remaining:05.1f}s", True, HUD_COLOR)
        surface.blit(txt, (20, 20))

        # Show phase indicator
        if self.thief.phase == Phase.TREASURE:
            phase_txt = self._font_small.render("CATCH THE THIEF!", True, (255, 200, 100))
        else:
            phase_txt = self._font_small.render("STOP THE ESCAPE!", True, (255, 100, 100))
        surface.blit(phase_txt, (GAME_W - phase_txt.get_width() - 20, 20))

    def _draw_end_screen(self, surface: pygame.Surface) -> None:
        veil = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 160))
        surface.blit(veil, (0, 0))
        if self.player_won:
            headline = self._font_large.render("THIEF CAUGHT!", True, (80, 200, 255))
            sub = self._font_med.render("Good aim. The treasure is safe.", True, (200, 235, 255))
        else:
            if self.thief.escaped:
                headline = self._font_large.render("THIEF ESCAPED!", True, (255, 80, 80))
                sub = self._font_med.render("The thief got away with the treasure...", True, (255, 200, 200))
            else:
                headline = self._font_large.render("TIME'S UP!", True, (255, 150, 50))
                sub = self._font_med.render("The thief disappeared into the shadows.", True, (255, 200, 150))
        restart = self._font_med.render("Press R to restart  |  Q to quit", True, (220, 220, 220))
        surface.blit(headline, (GAME_W // 2 - headline.get_width() // 2, GAME_H // 2 - 100))
        surface.blit(sub,      (GAME_W // 2 - sub.get_width() // 2,      GAME_H // 2))
        surface.blit(restart,  (GAME_W // 2 - restart.get_width() // 2,  GAME_H // 2 + 80))


def create(app):
    """Factory used by run.py's game registry."""
    return ThiefGame(app)
