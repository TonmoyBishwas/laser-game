"""Tuning dials for the fruit_ninja game.

All gameplay constants live here so the game logic stays readable and feel is
easy to adjust. Sizes are in the fixed 1920x1080 canvas space.
"""

from arcade.config import GAME_W, GAME_H, TARGET_FPS  # noqa: F401 (re-exported)

# Hit areas — kept generous because the real laser is shaky and can jitter.
FRUIT_RADIUS = 70          # px radius the swipe must pass within to slice a fruit
BOMB_RADIUS = 60           # px radius for the (smaller) bomb

# How big the sprites are drawn on screen.
FRUIT_DRAW = 150           # apple sprite size in px
BOMB_DRAW = 130            # bomb sprite size in px

# Physics.
GRAVITY = 400              # px / s^2 pulling everything back down

# Slicing: the pointer must move at least this fast (px per frame) to cut.
# A resting / slowly-aiming pointer must never slice — this is the gate for that.
MIN_SWIPE_SPEED = 22

# Lives. Missing a falling apple, or slicing a bomb, costs one.
START_LIVES = 3

# Slice feedback: how fast the two apple halves drift apart (px/s).
SPLIT_SPEED = 90

# Below this brightness a pixel is kept; at/above it the white photo background
# is made transparent so the apple isn't a white box on the wood.
WHITE_CUTOUT_THRESHOLD = 232
