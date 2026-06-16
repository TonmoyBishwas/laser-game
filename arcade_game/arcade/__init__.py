"""
arcade — the shared engine for laser-arcade.

This package is the boundary between your games and the real laser installation.
DO NOT edit it to add a game feature; build inside games/ instead (see CLAUDE.md).
The public pieces a game uses:

    from arcade.scene import Scene          # base class for a game
    from arcade.input import Pointer        # the only player input
    from arcade.dwell import DwellTracker   # hold-to-act (the laser's 'click')
    from arcade import gestures             # slice / swipe helpers
    from arcade.assets import AssetLoader   # load art + sound
    from arcade import palette              # safe (green-free) colors
    from arcade.config import GAME_W, GAME_H
"""

from arcade.config import GAME_H, GAME_W, TARGET_FPS  # noqa: F401
