"""
Global display + runtime constants for the whole arcade.

Every game draws onto a fixed 1920x1080 virtual canvas regardless of the actual
window size, so coordinates are identical in a small dev window and on the
full-size projector. Per-game tuning (sizes, speeds, colors) lives inside each
game folder, NOT here.
"""

# The virtual canvas every game draws onto. This matches the projector.
GAME_W, GAME_H = 1920, 1080

# Target frame rate for the main loop.
TARGET_FPS = 60

# Default windowed size used by `--windowed` (the canvas is scaled to fit this).
WINDOW_W, WINDOW_H = 1280, 720
