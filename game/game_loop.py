"""
Pygame game loop — maze + flashlight reveal + RL thief.

Render order each frame:
  1. Clear to black
  2. Blit cached maze surface (walls + spawn marker)
  3. Draw thief body
  4. Dark overlay with spotlight hole (reveals walls + thief in lit area)
  5. Treasure on top (always visible)
  6. Catch progress ring (above overlay)
  7. Edge glow (always visible — directional thief hint)
  8. HUD (timer, debug)
"""

import math
import time

import pygame

from game.config import (
    CATCH_HOLD_SECS,
    CATCH_RADIUS,
    CELL_H,
    CELL_W,
    EDGE_GLOW_BASE,
    EDGE_GLOW_DEPTH,
    EDGE_GLOW_PULSE,
    GAME_H,
    GAME_W,
    GRID_COLS,
    GRID_ROWS,
    LASER_STALE_SECS,
    ROUND_TIMEOUT_SECS,
    SPOTLIGHT_FADE_SECS,
    SPOTLIGHT_RADIUS,
    TARGET_FPS,
    TREASURE_H,
    TREASURE_W,
)
from game.maze import Maze
from game.maze_layouts import DEFAULT
from game.rl_env import NO_LASER
from game.rl_policy import QPolicy
from game.thief import Thief

# ── Colours ────────────────────────────────────────────────────────────────────
BG_COLOR          = (8, 8, 18)
WALL_COLOR        = (60, 160, 220)
WALL_INNER_COLOR  = (15, 60, 100)
TREASURE_COLOR    = (255, 215, 0)
TREASURE_GLOW     = (255, 240, 100)
HUD_COLOR         = (220, 220, 220)
OVERLAY_ALPHA_MAX = 245    # near-black when no laser; punched hole reveals area
EDGE_GLOW_COLOR   = (220, 50, 50)

# ── Fonts ──────────────────────────────────────────────────────────────────────
_FONT_LARGE = None
_FONT_MED   = None
_FONT_SMALL = None


def _init_fonts() -> None:
    global _FONT_LARGE, _FONT_MED, _FONT_SMALL
    _FONT_LARGE = pygame.font.SysFont("monospace", 72, bold=True)
    _FONT_MED   = pygame.font.SysFont("monospace", 36, bold=True)
    _FONT_SMALL = pygame.font.SysFont("monospace", 22)


# ── Spotlight surface cache ────────────────────────────────────────────────────
def _build_spotlight_surf(radius: int) -> pygame.Surface:
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        a = int(255 * (1 - r / radius) ** 0.6)
        pygame.draw.circle(surf, (255, 255, 200, a), (radius, radius), r)
    return surf


_SPOTLIGHT_BASE: pygame.Surface | None = None


def _get_spotlight_base() -> pygame.Surface:
    global _SPOTLIGHT_BASE
    if _SPOTLIGHT_BASE is None:
        _SPOTLIGHT_BASE = _build_spotlight_surf(SPOTLIGHT_RADIUS)
    return _SPOTLIGHT_BASE


# ── Maze surface cache ─────────────────────────────────────────────────────────
def _build_maze_surface(maze: Maze) -> pygame.Surface:
    surf = pygame.Surface((GAME_W, GAME_H))
    surf.fill(BG_COLOR)

    # Walls
    pad = 4
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if maze.is_wall(r, c):
                rect = pygame.Rect(c * CELL_W + pad, r * CELL_H + pad,
                                   CELL_W - 2 * pad, CELL_H - 2 * pad)
                pygame.draw.rect(surf, WALL_INNER_COLOR, rect)
                pygame.draw.rect(surf, WALL_COLOR, rect, width=3)
    return surf


# ── Overlay with spotlight hole ────────────────────────────────────────────────
def _draw_dark_overlay(
    surface: pygame.Surface,
    spot_x: float, spot_y: float,
    dark_alpha: int,
    hole_alpha: int,
) -> None:
    """
    Always darken the screen with `dark_alpha`. Punch a soft bright hole at
    (spot_x, spot_y) whose strength is `hole_alpha` (0=no hole, 255=full hole).
    """
    if dark_alpha <= 0:
        return
    overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, dark_alpha))

    if hole_alpha > 5:
        base = _get_spotlight_base()
        scaled = base.copy()
        # BLEND_RGBA_MULT scales every pixel's alpha by hole_alpha/255.
        scaled.fill((255, 255, 255, hole_alpha), special_flags=pygame.BLEND_RGBA_MULT)
        overlay.blit(
            scaled,
            (int(spot_x) - SPOTLIGHT_RADIUS, int(spot_y) - SPOTLIGHT_RADIUS),
            special_flags=pygame.BLEND_RGBA_SUB,
        )

    surface.blit(overlay, (0, 0))


# ── Treasure ───────────────────────────────────────────────────────────────────
def _draw_treasure(surface: pygame.Surface, elapsed: float, maze: Maze) -> None:
    cx, cy = maze.cell_center(*maze.treasure)
    rect = pygame.Rect(int(cx - TREASURE_W // 2), int(cy - TREASURE_H // 2),
                       TREASURE_W, TREASURE_H)
    pygame.draw.rect(surface, TREASURE_COLOR, rect, border_radius=10)

    pulse = int(abs(math.sin(elapsed * 2)) * 8)
    glow_rect = rect.inflate(pulse * 2, pulse * 2)
    pygame.draw.rect(surface, TREASURE_GLOW, glow_rect, width=3, border_radius=12)

    label = _FONT_SMALL.render("TREASURE", True, (40, 30, 0))
    surface.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom + 4))


# ── Edge glow (directional thief hint) ─────────────────────────────────────────
def _draw_edge_glow(surface: pygame.Surface, thief_pos: tuple[float, float], t: float) -> None:
    tx, ty = thief_pos
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

        if name == "top":
            strip = pygame.Surface((GAME_W, depth), pygame.SRCALPHA)
            for i in range(depth):
                a = int(max_alpha * (1 - i / depth))
                pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (0, i), (GAME_W, i))
            surface.blit(strip, (0, 0))
        elif name == "bottom":
            strip = pygame.Surface((GAME_W, depth), pygame.SRCALPHA)
            for i in range(depth):
                a = int(max_alpha * (1 - i / depth))
                pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (0, depth - 1 - i), (GAME_W, depth - 1 - i))
            surface.blit(strip, (0, GAME_H - depth))
        elif name == "left":
            strip = pygame.Surface((depth, GAME_H), pygame.SRCALPHA)
            for i in range(depth):
                a = int(max_alpha * (1 - i / depth))
                pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (i, 0), (i, GAME_H))
            surface.blit(strip, (0, 0))
        elif name == "right":
            strip = pygame.Surface((depth, GAME_H), pygame.SRCALPHA)
            for i in range(depth):
                a = int(max_alpha * (1 - i / depth))
                pygame.draw.line(strip, (*EDGE_GLOW_COLOR, a), (depth - 1 - i, 0), (depth - 1 - i, GAME_H))
            surface.blit(strip, (GAME_W - depth, 0))


# ── HUD ────────────────────────────────────────────────────────────────────────
def _draw_hud(surface: pygame.Surface, elapsed: float) -> None:
    remaining = max(0.0, ROUND_TIMEOUT_SECS - elapsed)
    txt = _FONT_MED.render(f"Time: {remaining:05.1f}s", True, HUD_COLOR)
    surface.blit(txt, (20, 20))


# ── End screens ────────────────────────────────────────────────────────────────
def _draw_end_screen(surface: pygame.Surface, player_won: bool) -> None:
    overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    if player_won:
        headline = _FONT_LARGE.render("THIEF CAUGHT!", True, (80, 255, 80))
        sub      = _FONT_MED.render("Good aim. The treasure is safe.", True, (200, 255, 200))
    else:
        headline = _FONT_LARGE.render("THIEF ESCAPED!", True, (255, 80, 80))
        sub      = _FONT_MED.render("The treasure is gone...", True, (255, 200, 200))

    restart = _FONT_MED.render("Press R to restart  |  Q to quit", True, (220, 220, 220))
    surface.blit(headline, (GAME_W // 2 - headline.get_width() // 2, GAME_H // 2 - 100))
    surface.blit(sub,      (GAME_W // 2 - sub.get_width() // 2,      GAME_H // 2))
    surface.blit(restart,  (GAME_W // 2 - restart.get_width() // 2,  GAME_H // 2 + 80))


# ── Debug HUD ──────────────────────────────────────────────────────────────────
def _draw_debug_hud(
    surface: pygame.Surface,
    shared,
    spot_pos, spot_alpha: float, laser_fresh: bool,
    thief: Thief,
    policy: QPolicy,
) -> None:
    stats = shared.detector_stats
    with shared.laser_lock:
        ls = shared.laser_state

    lines = [
        f"DETECTOR     infer={stats.infer_fps:5.1f} fps   detect={stats.detect_fps:5.1f} fps",
        f"TOTALS       frames={stats.total_frames}   detections={stats.total_detections}",
    ]
    if ls is None:
        lines.append("LASER        NEVER DETECTED")
    else:
        age_ms = (time.time() - ls.timestamp) * 1000
        gx, gy = ls.pos
        lines.append(f"LASER        cam=({ls.camera_pos[0]:6.1f},{ls.camera_pos[1]:6.1f})  game=({gx:7.1f},{gy:7.1f})")
        lines.append(f"             conf={ls.conf:.2f}  age={age_ms:5.0f} ms  fresh={laser_fresh}")
    lines.append(
        f"THIEF        cell={thief.cell}  next={thief.next_cell}  catch={thief.catch_progress:.2f}"
    )

    panel = pygame.Surface((760, 24 * len(lines) + 16), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    surface.blit(panel, (GAME_W - 780, GAME_H - panel.get_height() - 20))
    for i, line in enumerate(lines):
        txt = _FONT_SMALL.render(line, True, (200, 255, 200))
        surface.blit(txt, (GAME_W - 768, GAME_H - panel.get_height() - 12 + i * 24))

    # Q-value bars for current state (bottom-left)
    state = (
        thief.cell[0], thief.cell[1],
        ls.pos[1] // CELL_H if ls is not None else NO_LASER[0],
        ls.pos[0] // CELL_W if ls is not None else NO_LASER[1],
    )
    state = tuple(int(v) for v in state)
    q_vals = policy.values(state)
    q_min, q_max = float(q_vals.min()), float(q_vals.max())
    rng = max(0.01, q_max - q_min)

    bar_x, bar_y = 20, GAME_H - 220
    bar_w_max = 200
    bar_h = 24
    actions = ["N", "E", "S", "W"]
    title = _FONT_SMALL.render(f"Q-values  state=({state[0]},{state[1]} | {state[2]},{state[3]})", True, (220, 220, 100))
    surface.blit(title, (bar_x, bar_y - 28))
    best_a = int(q_vals.argmax())
    for i, name in enumerate(actions):
        v = float(q_vals[i])
        norm = (v - q_min) / rng
        w = int(bar_w_max * norm)
        color = (80, 255, 80) if i == best_a else (140, 140, 180)
        pygame.draw.rect(surface, (40, 40, 60), (bar_x + 40, bar_y + i * (bar_h + 4), bar_w_max, bar_h))
        pygame.draw.rect(surface, color,         (bar_x + 40, bar_y + i * (bar_h + 4), max(2, w), bar_h))
        lbl = _FONT_SMALL.render(f"{name} {v:+.2f}", True, (220, 220, 220))
        surface.blit(lbl, (bar_x, bar_y + i * (bar_h + 4)))

    # Always-visible thief silhouette
    thief.draw_silhouette(surface)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(shared, policy: QPolicy, fullscreen: bool = False, debug: bool = False) -> None:
    pygame.init()
    _init_fonts()

    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((GAME_W, GAME_H), flags)
    pygame.display.set_caption("Laser Guard — find the thief!")
    clock = pygame.time.Clock()

    maze = Maze(DEFAULT)
    maze_surface = _build_maze_surface(maze)
    thief = Thief(maze, policy)

    elapsed = 0.0
    game_over   = False
    player_won  = False

    spot_alpha = 0.0
    spot_pos   = (GAME_W / 2, GAME_H / 2)
    fade_rate  = 255.0 / max(SPOTLIGHT_FADE_SECS, 0.01)

    print(f"[game] starting — maze sig={maze.signature}, debug={debug}")

    while True:
        dt = clock.tick(TARGET_FPS) / 1000.0

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if event.key == pygame.K_r and game_over:
                    thief.reset()
                    elapsed = 0.0
                    game_over  = False
                    player_won = False
                    spot_alpha = 0.0

        if game_over:
            screen.blit(maze_surface, (0, 0))
            _draw_treasure(screen, elapsed, maze)
            thief.draw(screen)
            _draw_end_screen(screen, player_won)
            pygame.display.flip()
            continue

        # ── Laser ─────────────────────────────────────────────────────────────
        with shared.laser_lock:
            ls = shared.laser_state

        laser_fresh = (ls is not None) and ((time.time() - ls.timestamp) < LASER_STALE_SECS)
        if laser_fresh:
            spot_pos   = ls.pos
            spot_alpha = min(255.0, spot_alpha + fade_rate * dt * 4)
        else:
            spot_alpha = max(0.0, spot_alpha - fade_rate * dt)
        laser_active = spot_alpha > 128

        # Convert laser pixel → maze cell (for the policy)
        laser_cell = None
        if laser_fresh:
            cr, cc = maze.pixel_to_cell(spot_pos[0], spot_pos[1])
            laser_cell = (cr, cc)

        # ── Update state ──────────────────────────────────────────────────────
        elapsed += dt
        thief.update(dt, spot_pos if laser_active else None, laser_active, laser_cell)

        if thief.caught:
            game_over = True
            player_won = True
        elif thief.reached:
            game_over = True
            player_won = False
        elif elapsed >= ROUND_TIMEOUT_SECS:
            game_over = True
            player_won = False

        # ── Render ────────────────────────────────────────────────────────────
        screen.blit(maze_surface, (0, 0))
        thief.draw(screen)

        # Dark overlay (constant); the spotlight hole's strength fades with spot_alpha
        _draw_dark_overlay(
            screen,
            spot_pos[0], spot_pos[1],
            dark_alpha=OVERLAY_ALPHA_MAX,
            hole_alpha=int(spot_alpha),
        )

        # Treasure ALWAYS visible
        _draw_treasure(screen, elapsed, maze)

        # Catch progress ring (above overlay so always visible when active)
        thief.draw_catch_ring(screen)

        # Edge glow direction hint
        _draw_edge_glow(screen, thief.pos, elapsed)

        # HUD
        _draw_hud(screen, elapsed)
        if debug:
            _draw_debug_hud(screen, shared, spot_pos, spot_alpha, laser_fresh, thief, policy)

        pygame.display.flip()
