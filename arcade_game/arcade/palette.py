"""
Color helpers + the single hard color rule of this project.

THE RULE: bright/saturated GREEN is reserved for the laser dot. Art and UI must
never use it, or the real detector confuses the picture with the laser. Everything
in here helps you stay on the safe side of that line. Full explanation lives in
docs/ASSET_GUIDE.md.
"""

from __future__ import annotations

# A reference for the color you must avoid filling the screen with.
LASER_GREEN = (40, 255, 70)

# A ready-made green-free palette. Reach for these instead of inventing greens.
WHITE      = (240, 240, 245)
BLACK      = (8, 8, 18)
GOLD       = (255, 215, 0)
AMBER      = (255, 170, 40)
ORANGE     = (255, 120, 40)
RED        = (220, 40, 40)
CRIMSON    = (140, 0, 0)
PINK       = (255, 90, 140)
MAGENTA    = (220, 60, 200)
PURPLE     = (150, 80, 230)
BLUE       = (60, 120, 230)
CYAN       = (80, 200, 255)
TEAL       = (40, 160, 170)   # blue-leaning teal — safe; a pure green is not
SLATE      = (60, 80, 110)


def is_reserved_green(rgb: tuple[int, int, int]) -> bool:
    """
    True if a color sits in the 'laser green' danger zone: green clearly dominant
    over red and blue, and bright. Muted blue-greens (teal) and dark colors pass.
    Use this to lint art-derived colors, not as a precise science.
    """
    r, g, b = rgb[0], rgb[1], rgb[2]
    return g > 150 and r < 130 and b < 130 and (g - max(r, b)) > 60


def warn_if_green(rgb: tuple[int, int, int], where: str = "") -> bool:
    """Print a warning (and return True) if a color breaks the green rule."""
    if is_reserved_green(rgb):
        tag = f" [{where}]" if where else ""
        print(f"[palette] WARNING{tag}: {rgb} is reserved 'laser green' — "
              f"pick a non-green color (see docs/ASSET_GUIDE.md).")
        return True
    return False
