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

# Thief sprite sheet (Thief.png): a row of 3 walking poses, animated while moving.
THIEF_FRAMES   = 3      # number of poses in the sheet
THIEF_DRAW_H   = 118    # on-screen height (px); width follows the sprite's aspect
THIEF_ANIM_FPS = 8      # walk-cycle speed (frames per second) while moving

# ── Door (escape target) ───────────────────────────────────────────────────────────
DOOR_W, DOOR_H = 104, 90             # On-screen size (cave-mouth art reads larger)
DOOR_DRAW_W, DOOR_DRAW_H = 140, 120  # Artist reference (2x for crisp scaling)

# ── Escape phase (after thief gets treasure) ────────────────────────────────────────
ESCAPE_SPEED_MULT = 0.72  # Multiplier for CELL_TRANSITION_SECS (lower = faster).
                          # Softened from 0.6 so the long run to a far door stays
                          # catchable. This is the PRIMARY difficulty dial: lower it
                          # toward 0.6 for a harder thief, raise it for an easier one.

# ── Escape doors (the exits the thief races for) ────────────────────────────────────
# Exactly two doors are placed far from the treasure and far from each other, so
# guarding one never covers the other. While escaping, the thief picks whichever
# door is currently safer; if the player guards a door (laser on it or its route),
# the thief heads for the other one — then commits once past the fork between them.
NUM_DOORS              = 2     # how many escape doors to place
# Door spacing is measured as PHYSICAL grid distance (Manhattan), not path distance:
# the player sees a door's on-screen position, so two doors on opposite sides of a
# wall (long path, but touching on screen) must still count as "too close".
DOOR_MIN_TREASURE_DIST = 6     # min grid cells between the treasure and each door
DOOR_MIN_PAIR_DIST     = 6     # min grid cells between the two doors (so one spotlight
                               # can't cover both and the player must commit/chase)
DOOR_GUARD_RADIUS      = 2     # laser within this many cells of a door => "guarded"
DOOR_GUARD_PENALTY     = 50    # score penalty for a guarded door (finite, so two
                               # guarded doors still compare instead of both = inf)
DOOR_ROUTE_RADIUS      = 1     # laser this close to the route ahead => "blocked"
DOOR_ROUTE_PENALTY     = 8     # per-cell-closeness penalty for a blocked route
DOOR_SWITCH_HYSTERESIS = 6     # a rival door must beat the current one by MORE than
                               # this to trigger a switch (prevents flip-flop)
DOOR_COMMIT_DIST       = 3     # within this many cells of a door, lock the choice

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
