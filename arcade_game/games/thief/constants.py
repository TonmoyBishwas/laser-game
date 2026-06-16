"""
Tuning constants for the Thief game.

These are the dials you can turn to change how the game feels. They mirror the
real installation's config so changes here port straight back. GAME_W/GAME_H and
TARGET_FPS come from the shared engine so there is a single source of truth.
"""

from arcade.config import GAME_H, GAME_W, TARGET_FPS  # noqa: F401

# ── Spotlight (the flashlight the pointer reveals the maze with) ────────────────
SPOTLIGHT_RADIUS    = 160     # px — how much the pointer reveals
SPOTLIGHT_FADE_SECS = 0.3     # how quickly the light fades in/out

# ── Maze grid ───────────────────────────────────────────────────────────────────
GRID_COLS, GRID_ROWS = 15, 9
CELL_W = GAME_W // GRID_COLS   # 128
CELL_H = GAME_H // GRID_ROWS   # 120
CELL_TRANSITION_SECS = 1.20    # seconds for the thief to slide one cell

# ── Thief + treasure sizes ───────────────────────────────────────────────────────
THIEF_W, THIEF_H = 60, 80
TREASURE_W, TREASURE_H = 90, 70

# ── Door (escape target) ───────────────────────────────────────────────────────────
DOOR_W, DOOR_H = 70, 60              # On-screen size
DOOR_DRAW_W, DOOR_DRAW_H = 140, 120  # Artist reference (2x for crisp scaling)

# ── Escape phase (after thief gets treasure) ────────────────────────────────────────
ESCAPE_SPEED_MULT = 0.6  # Multiplier for CELL_TRANSITION_SECS (lower = faster)
                        # 0.6 means 60% of normal time = ~1.67x faster

# ── Catch mechanic (hold the pointer on the thief) ───────────────────────────────
CATCH_HOLD_SECS   = 1.0       # how long the pointer must stay on the thief
CATCH_RADIUS      = 100       # px — slightly smaller than SPOTLIGHT_RADIUS
CATCH_DECAY_RATIO = 0.5       # how fast progress drains when the pointer leaves

# ── Edge glow (hint pointing toward the thief) ───────────────────────────────────
EDGE_GLOW_DEPTH = 60
EDGE_GLOW_BASE  = 0.45
EDGE_GLOW_PULSE = 0.55

# ── Round timing ─────────────────────────────────────────────────────────────────
ROUND_TIMEOUT_SECS = 90       # round resets if nobody wins by then

# Sentinel meaning "no pointer this frame" in the thief's policy state.
NO_LASER = (-1, -1)
