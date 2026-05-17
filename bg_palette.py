"""
Shared background-colour palette used by make_bg_images.py and live_bg_test.py.

30 colours total. The first 10 are the original EIZO-style primaries kept for
back-compat with earlier results. Entries 11-30 lean dark / non-red so the
bright-red laser dot has the highest contrast against the projected surface.

Format: list of (name, (R, G, B)).
"""

COLORS = [
    # ── original 10 (keep order so old leaderboards still align) ───────────────
    ("01_black",        (  0,   0,   0)),
    ("02_white",        (255, 255, 255)),
    ("03_red",          (255,   0,   0)),
    ("04_green",        (  0, 255,   0)),
    ("05_blue",         (  0,   0, 255)),
    ("06_cyan",         (  0, 255, 255)),
    ("07_magenta",      (255,   0, 255)),
    ("08_yellow",       (255, 255,   0)),
    ("09_gray50",       (128, 128, 128)),
    ("10_gray25",       ( 64,  64,  64)),

    # ── darker greys ──────────────────────────────────────────────────────────
    ("11_gray37",       ( 96,  96,  96)),
    ("12_gray12",       ( 32,  32,  32)),
    ("13_gray06",       ( 16,  16,  16)),
    ("14_charcoal",     ( 40,  40,  40)),

    # ── dark blues (red laser pops against blue) ──────────────────────────────
    ("15_dark_blue",    (  0,   0,  64)),
    ("16_navy",         (  0,   0, 128)),
    ("17_midnight",     ( 16,  16,  32)),
    ("18_indigo",       ( 32,   0,  64)),
    ("19_blue_gray",    ( 32,  48,  64)),
    ("20_slate",        ( 48,  56,  64)),

    # ── dark greens ───────────────────────────────────────────────────────────
    ("21_dark_green",   (  0,  64,   0)),
    ("22_forest",       (  0,  80,   0)),
    ("23_forest_dark",  ( 12,  36,  12)),
    ("24_deep_green",   (  0,  96,  32)),

    # ── dark cyans / teals ────────────────────────────────────────────────────
    ("25_dark_cyan",    (  0,  64,  64)),
    ("26_teal",         (  0,  96,  96)),

    # ── dark magentas / purples ───────────────────────────────────────────────
    ("27_dark_magenta", ( 64,   0,  64)),
    ("28_purple",       ( 48,   0,  96)),

    # ── dark yellows / olives ─────────────────────────────────────────────────
    ("29_olive",        ( 64,  64,   0)),
    ("30_dark_olive",   ( 40,  40,   0)),
]
