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
import random
from pathlib import Path
from typing import Any

import pygame

from arcade.assets import AssetLoader
from arcade.scene import Scene

from .constants import (
    CELL_H,
    CELL_W,
    DOOR_H,
    DOOR_MIN_PAIR_DIST,
    DOOR_MIN_TREASURE_DIST,
    DOOR_W,
    EDGE_GLOW_BASE,
    EDGE_GLOW_DEPTH,
    EDGE_GLOW_PULSE,
    GAME_H,
    GAME_W,
    GRID_COLS,
    GRID_ROWS,
    NUM_DOORS,
    ROUND_TIMEOUT_SECS,
    SPOTLIGHT_FADE_SECS,
    SPOTLIGHT_RADIUS,
    THIEF_DRAW_H,
    THIEF_FRAMES,
    TREASURE_H,
    TREASURE_W,
)
from .maze import Maze
from .maze_layouts import DEFAULT, random_layout
from .pathfinding import bfs_distances
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
        self.treasure_thief_img: pygame.Surface | None = None
        self.wall_img: pygame.Surface | None = None
        self.thief_frames: list[pygame.Surface] = []
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
        self.num_doors: int = NUM_DOORS  # Number of escape doors to place

        self._spotlight = _build_spotlight_surf(SPOTLIGHT_RADIUS)
        self._overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)

        # ── Per-frame scratch surfaces, allocated ONCE and reused. Allocating
        # full-screen (or even spotlight-sized) surfaces inside draw() every
        # frame is main-thread work that starves the detector thread in the
        # real installation (see DETECTION_POSTMORTEM.md there).
        self._shake_surf = pygame.Surface((GAME_W, GAME_H))
        self._flash_surf = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        self._spot_scratch = pygame.Surface(self._spotlight.get_size(), pygame.SRCALPHA)

        # Static text rendered once (SysFont construction per frame is very
        # expensive — the old ESCAPE! banner made 2 fonts + 37 renders a frame).
        self._escape_banner = self._build_escape_banner()
        self._escape_sub = pygame.font.SysFont("monospace", 28, bold=True).render(
            "Find the door! Catch the thief!", True, (255, 200, 100))
        self._phase_txts = {
            True:  self._font_small.render("CATCH THE THIEF!", True, (255, 200, 100)),
            False: self._font_small.render("STOP THE ESCAPE!", True, (255, 100, 100)),
        }
        self._veil = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        self._veil.fill((0, 0, 0, 160))
        self._end_cache: tuple | None = None   # (headline, sub, restart) built on game over

        # Edge-glow gradient strips pre-rendered at full alpha; per frame we
        # just set_alpha + blit (was: 4 surface allocs + ~240 draw.line calls).
        self._edge_strips = self._build_edge_strips()

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

        # The bare treasure the thief carries away during the escape.
        try:
            self.treasure_thief_img = self._assets.load_image("treasure_thief.png", size=(TREASURE_W, TREASURE_H))
        except Exception:
            self.treasure_thief_img = None

        # Wall tile — drawn once per wall cell, sized to fill a grid cell exactly.
        try:
            self.wall_img = self._assets.load_image("wall_tile.png", size=(CELL_W, CELL_H))
        except Exception:
            self.wall_img = None  # Fallback to drawn rectangles

        # Thief walking sprite sheet — a row of THIEF_FRAMES poses, sliced + scaled.
        self.thief_frames = self._load_thief_frames()

    def _load_thief_frames(self) -> list[pygame.Surface]:
        """Slice Thief.png (a row of THIEF_FRAMES poses) into scaled walk frames."""
        try:
            sheet = self._assets.load_image("Thief.png")  # full sheet, unscaled
            sw, sh = sheet.get_size()
            fw = sw // THIEF_FRAMES
            scale = THIEF_DRAW_H / sh
            draw_w = max(1, int(fw * scale))
            frames = []
            for i in range(THIEF_FRAMES):
                sub = sheet.subsurface(pygame.Rect(i * fw, 0, fw, sh)).copy()
                frames.append(pygame.transform.smoothscale(sub, (draw_w, THIEF_DRAW_H)))
            return frames
        except Exception:
            return []  # Fallback: thief draws its placeholder rectangle

    def _build_escape_banner(self) -> pygame.Surface:
        """Pre-render the pulsing "ESCAPE!" banner ONCE at its largest size
        (pulse max = 1.2 -> 86pt) with the black outline baked in. draw() just
        smoothscales this small surface to the current pulse size — the old
        code built two SysFonts and did 37 text renders EVERY frame."""
        font = pygame.font.SysFont("monospace", int(72 * 1.2), bold=True)
        text = font.render("ESCAPE!", True, (255, 80, 80))
        outline = font.render("ESCAPE!", True, (0, 0, 0))
        pad = 3
        banner = pygame.Surface((text.get_width() + 2 * pad,
                                 text.get_height() + 2 * pad), pygame.SRCALPHA)
        for dx in (-3, -2, -1, 1, 2, 3):
            for dy in (-3, -2, -1, 1, 2, 3):
                banner.blit(outline, (pad + dx, pad + dy))
        banner.blit(text, (pad, pad))
        return banner

    def _build_edge_strips(self) -> dict[str, pygame.Surface]:
        """Pre-render the four edge-glow gradient strips at full alpha."""
        depth = EDGE_GLOW_DEPTH
        strips: dict[str, pygame.Surface] = {}
        for name in ("top", "bottom", "left", "right"):
            if name in ("top", "bottom"):
                strip = pygame.Surface((GAME_W, depth), pygame.SRCALPHA)
                for i in range(depth):
                    a = int(255 * (1 - i / depth))
                    row = i if name == "top" else depth - 1 - i
                    pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (0, row), (GAME_W, row))
            else:
                strip = pygame.Surface((depth, GAME_H), pygame.SRCALPHA)
                for i in range(depth):
                    a = int(255 * (1 - i / depth))
                    col = i if name == "left" else depth - 1 - i
                    pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (col, 0), (col, GAME_H))
            strips[name] = strip
        return strips

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

        Spawn/treasure come from the 'S'/'T' markers (or random open cells if a grid
        somehow lacks them). Doors are always (re)computed by `_far_doors` so they sit
        far from the treasure and far from each other — any 'D' markers in the grid are
        ignored (they stay as ordinary walkable cells).
        """
        spawn = None
        treasure = None
        open_cells = []

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                val = grid[r][c]
                if val == 'S':
                    spawn = (r, c)
                elif val == 'T':
                    treasure = (r, c)
                elif val == 0:
                    open_cells.append((r, c))

        # If no explicit markers, pick random positions
        if spawn is None and open_cells:
            if len(open_cells) >= 2:
                spawn, treasure = random.sample(open_cells, 2)
            else:
                spawn, treasure = open_cells[0], open_cells[0]

        # Fallback to defaults if still None
        if spawn is None:
            spawn = (GRID_ROWS - 2, GRID_COLS // 2)
        if treasure is None:
            treasure = (1, GRID_COLS // 2)

        doors = self._far_doors(grid, spawn, treasure)
        return spawn, treasure, doors

    def _far_doors(self, grid: list[list[Any]], spawn: tuple[int, int],
                   treasure: tuple[int, int]) -> list[tuple[int, int]]:
        """Place two escape doors physically far from the treasure and from each other.

        Distances are PHYSICAL grid (Manhattan) distance — what the player sees on
        screen — so two doors separated by a wall but touching on screen still count as
        too close. BFS is used only to confirm a cell is reachable (a path exists);
        unreachable cells can't be escape doors. Thresholds relax progressively so even
        small/open mazes still yield two well-separated doors.
        """
        def in_bounds(r: int, c: int) -> bool:
            return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS

        def is_wall(r: int, c: int) -> bool:
            return (not in_bounds(r, c)) or grid[r][c] == 1

        def degree(cell: tuple[int, int]) -> int:
            r, c = cell
            return sum(1 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if in_bounds(r + dr, c + dc) and not is_wall(r + dr, c + dc))

        def mh(a: tuple[int, int], b: tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        reachable = [c for c in bfs_distances(treasure, is_wall, in_bounds)
                     if c not in (spawn, treasure)]
        if len(reachable) < 2:
            return reachable or [spawn]

        # Try strict spacing first; relax both gates if the maze is too cramped.
        for min_t, min_pair in ((DOOR_MIN_TREASURE_DIST, DOOR_MIN_PAIR_DIST),
                                 (DOOR_MIN_TREASURE_DIST - 2, DOOR_MIN_PAIR_DIST - 2),
                                 (3, 4), (2, 3), (1, 1)):
            cand = [c for c in reachable if mh(c, treasure) >= min_t and degree(c) >= 2]
            if len(cand) < 2:
                cand = [c for c in reachable if mh(c, treasure) >= min_t]  # allow dead-ends
            if len(cand) < 2:
                continue
            # Door A: farthest from the treasure. Door B: far from BOTH the treasure and
            # door A (maximize the smaller gap), and at least min_pair from door A.
            door_a = max(cand, key=lambda c: (mh(c, treasure), c))
            bpool = [c for c in cand if c != door_a and mh(c, door_a) >= min_pair]
            if not bpool:
                continue
            door_b = max(bpool, key=lambda c: (min(mh(c, treasure), mh(c, door_a)), c))
            return [door_a, door_b]

        # Fallback: the reachable pair that is most spread out overall.
        best, best_gap = None, -1
        for i, a in enumerate(reachable):
            for b in reachable[i + 1:]:
                gap = min(mh(a, treasure), mh(b, treasure), mh(a, b))
                if gap > best_gap:
                    best, best_gap = (a, b), gap
        return list(best) if best else reachable[:2]

    def _build_maze_surface(self) -> pygame.Surface:
        surf = pygame.Surface((GAME_W, GAME_H))
        # Use background image if available, otherwise fallback to solid color
        if self.background_img is not None:
            surf.blit(self.background_img, (0, 0))
        else:
            surf.fill(BG_COLOR)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.maze.is_wall(r, c):
                    if self.wall_img is not None:
                        surf.blit(self.wall_img, (c * CELL_W, r * CELL_H))
                    else:
                        pad = 4
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

        # Create thief with spawn position, and give it the walking sprite frames.
        self.thief = Thief(self.maze, spawn_pos=spawn)
        self.thief.walk_frames = self.thief_frames

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
        self._end_cache = None

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
        # Freeze the thief while the capture flash is playing, so it doesn't
        # dart away under cover of the screen flash. It resumes once the flash
        # has fully faded.
        if self.capture_flash_alpha <= 0:
            self.thief.update(dt, self.spot_pos if laser_active else None, laser_active, laser_cell)

        # Check for treasure reached - switch to escape phase
        if self.thief.reached_treasure and not self.door_visible:
            self.door_visible = True
            self.treasure_captured_time = self.elapsed  # Track capture time for effects
            self.capture_flash_alpha = 255.0  # Start flash effect
            self.screen_shake = 10.0  # Screen shake magnitude

            # Hand the thief every door it can actually reach and let it choose
            # adaptively (see Thief.set_escape_doors). Random mazes only guarantee
            # spawn<->treasure connectivity, so a door can be walled off; if none are
            # reachable, fall back to spawn, which always is.
            reachable = [d for d in self.door_positions if self.thief.can_reach(d)]
            if reachable:
                self.thief.set_escape_doors(reachable)
            else:
                self.thief.set_escape_goal(self.spawn_pos)
            self.active_escape_door = self.thief.escape_goal

        # Update visual effects
        if self.capture_flash_alpha > 0:
            self.capture_flash_alpha -= dt * 150  # Fade out flash
            if self.capture_flash_alpha < 0:
                self.capture_flash_alpha = 0

        if self.screen_shake > 0:
            self.screen_shake -= dt * 20  # Reduce shake
            if self.screen_shake < 0:
                self.screen_shake = 0

        # Keep the displayed active door in sync with the thief's live choice.
        if self.door_visible and self.thief.escape_goal is not None:
            self.active_escape_door = self.thief.escape_goal

        # End conditions
        if self.thief.caught:
            self.game_over, self.player_won = True, True
        elif self.thief.escaped:
            self.game_over, self.player_won = True, False
        elif self.elapsed >= ROUND_TIMEOUT_SECS:
            self.game_over, self.player_won = True, False

    # ── Draw ────────────────────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface) -> None:
        # While shaking, compose the scene onto the reused scratch surface and
        # blit it at a jittered offset; otherwise draw straight to the canvas.
        # (One shared draw path — the old code duplicated it per branch and
        # allocated a fresh 1920x1080 surface every shake frame.)
        shaking = self.screen_shake > 0
        target = self._shake_surf if shaking else surface

        target.blit(self.maze_surface, (0, 0))
        self.thief.draw(target)
        # Draw stolen treasure on thief during escape
        if self.door_visible:
            self.thief.draw_stolen_treasure(target, self.treasure_thief_img)
        # Treasure is drawn BEFORE the darkness overlay, so it stays hidden in
        # the dark and is only revealed by the spotlight — like the walls/thief.
        self._draw_treasure(target)
        self._draw_overlay(target)
        self._draw_door(target)
        self.thief.draw_catch_ring(target)
        self._draw_edge_glow(target)
        self._draw_hud(target)

        if shaking:
            shake_x = random.uniform(-self.screen_shake, self.screen_shake)
            shake_y = random.uniform(-self.screen_shake, self.screen_shake)
            surface.blit(target, (shake_x, shake_y))

        # Draw treasure capture flash effect (reused surface, refilled per frame)
        if self.capture_flash_alpha > 0:
            self._flash_surf.fill((255, 100, 50, int(self.capture_flash_alpha)))
            surface.blit(self._flash_surf, (0, 0))

        # Draw "ESCAPE!" announcement — pre-rendered banner, scaled to the pulse
        if self.door_visible and not self.game_over:
            time_since_capture = self.elapsed - (self.treasure_captured_time or 0)
            if time_since_capture < 3.0:  # Show for 3 seconds
                pulse = 1.0 + 0.2 * abs(math.sin(self.elapsed * 5))
                scale = pulse / 1.2   # banner is pre-rendered at pulse max 1.2
                w = max(1, int(self._escape_banner.get_width() * scale))
                h = max(1, int(self._escape_banner.get_height() * scale))
                banner = pygame.transform.smoothscale(self._escape_banner, (w, h))
                surface.blit(banner, banner.get_rect(center=(GAME_W // 2, 100)))
                surface.blit(self._escape_sub,
                             self._escape_sub.get_rect(center=(GAME_W // 2, 160)))

        if self.game_over:
            self._draw_end_screen(surface)

    def _draw_overlay(self, surface: pygame.Surface) -> None:
        self._overlay.fill((0, 0, 0, OVERLAY_ALPHA_MAX))
        hole_alpha = int(self.spot_alpha)
        if hole_alpha > 5:
            # Rebuild the alpha-modulated spotlight in a reused scratch surface
            # (zero-fill + additive blit copies it exactly) instead of .copy(),
            # which allocated a new spotlight-sized surface every frame.
            scratch = self._spot_scratch
            scratch.fill((0, 0, 0, 0))
            scratch.blit(self._spotlight, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            if hole_alpha < 255:
                scratch.fill((255, 255, 255, hole_alpha), special_flags=pygame.BLEND_RGBA_MULT)
            self._overlay.blit(
                scratch,
                (int(self.spot_pos[0]) - SPOTLIGHT_RADIUS, int(self.spot_pos[1]) - SPOTLIGHT_RADIUS),
                special_flags=pygame.BLEND_RGBA_SUB,
            )
        surface.blit(self._overlay, (0, 0))

    def _draw_treasure(self, surface: pygame.Surface) -> None:
        cx, cy = self.maze.cell_center(*self.maze.treasure)
        rect = pygame.Rect(int(cx - TREASURE_W // 2), int(cy - TREASURE_H // 2),
                           TREASURE_W, TREASURE_H)

        # After the theft, the pedestal shows the empty display case.
        if self.door_visible:
            if self.treasure_stolen_img is not None:
                surface.blit(self.treasure_stolen_img, rect.topleft)
            return

        if self.treasure_img is not None:
            # Just the sprite — no glow, no label. The darkness hides it until the
            # spotlight sweeps over it, exactly like every other object in the maze.
            surface.blit(self.treasure_img, rect.topleft)
        else:
            # Fallback to a plain drawn treasure (no pulsing glow / label).
            pygame.draw.rect(surface, TREASURE_COLOR, rect, border_radius=10)

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

            # Both doors are drawn identically: the thief's real target is hidden, so
            # the player must read its movement rather than follow a telegraphed arrow.
            rect = pygame.Rect(int(cx - DOOR_W // 2), int(cy - DOOR_H // 2), DOOR_W, DOOR_H)

            if self.door_img is not None:
                # Soft elliptical halo behind the cave mouth — brighter the closer the
                # spotlight centre is, with a gentle pulse. No hard rectangle/handle,
                # which would look wrong on the rounded cave art.
                pulse = 0.6 + 0.4 * abs(math.sin(self.elapsed * 3))
                halo_alpha = int(90 * glow_intensity * pulse)
                if halo_alpha > 4:
                    halo = rect.inflate(40, 30)
                    halo_surf = pygame.Surface((halo.w, halo.h), pygame.SRCALPHA)
                    pygame.draw.ellipse(halo_surf, (255, 200, 110, halo_alpha), halo_surf.get_rect())
                    surface.blit(halo_surf, halo.topleft)
                surface.blit(self.door_img, rect.topleft)
            else:
                # Fallback placeholder if door.png is missing.
                pygame.draw.rect(surface, (139, 90, 50), rect, border_radius=5)
                pygame.draw.rect(surface, (100, 60, 30), rect, width=3, border_radius=5)
                glow_size = int(8 * glow_intensity)
                pygame.draw.rect(surface, (255, 200, 100),
                                 rect.inflate(glow_size * 2, glow_size * 2),
                                 width=2, border_radius=8)
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
            # Pre-rendered gradient strip; surface alpha scales the whole
            # gradient (multiplies with the per-pixel alpha in pygame 2).
            strip = self._edge_strips[name]
            strip.set_alpha(max_alpha)
            if name == "top":
                surface.blit(strip, (0, 0))
            elif name == "bottom":
                surface.blit(strip, (0, GAME_H - depth))
            elif name == "left":
                surface.blit(strip, (0, 0))
            else:
                surface.blit(strip, (GAME_W - depth, 0))

    def _draw_hud(self, surface: pygame.Surface) -> None:
        remaining = max(0.0, ROUND_TIMEOUT_SECS - self.elapsed)
        txt = self._font_med.render(f"Time: {remaining:05.1f}s", True, HUD_COLOR)
        surface.blit(txt, (20, 20))

        # Show phase indicator (pre-rendered — it only has two states)
        phase_txt = self._phase_txts[self.thief.phase == Phase.TREASURE]
        surface.blit(phase_txt, (GAME_W - phase_txt.get_width() - 20, 20))

    def _draw_end_screen(self, surface: pygame.Surface) -> None:
        surface.blit(self._veil, (0, 0))
        # Render the end-screen text once when the game-over state is first
        # drawn, then reuse (this screen shows for many frames).
        if self._end_cache is None:
            if self.player_won:
                headline = self._font_large.render("THIEF CAUGHT!", True, (80, 200, 255))
                sub = self._font_med.render("Good aim. The treasure is safe.", True, (200, 235, 255))
            elif self.thief.escaped:
                headline = self._font_large.render("THIEF ESCAPED!", True, (255, 80, 80))
                sub = self._font_med.render("The thief got away with the treasure...", True, (255, 200, 200))
            else:
                headline = self._font_large.render("TIME'S UP!", True, (255, 150, 50))
                sub = self._font_med.render("The thief disappeared into the shadows.", True, (255, 200, 150))
            restart = self._font_med.render("Press R to restart  |  Q to quit", True, (220, 220, 220))
            self._end_cache = (headline, sub, restart)
        headline, sub, restart = self._end_cache
        surface.blit(headline, (GAME_W // 2 - headline.get_width() // 2, GAME_H // 2 - 100))
        surface.blit(sub,      (GAME_W // 2 - sub.get_width() // 2,      GAME_H // 2))
        surface.blit(restart,  (GAME_W // 2 - restart.get_width() // 2,  GAME_H // 2 + 80))


def create(app):
    """Factory used by run.py's game registry."""
    return ThiefGame(app)
