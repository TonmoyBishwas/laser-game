"""
Hand-designed maze layouts.

Cell encoding:
  0 = open path
  1 = wall
  'S' = thief spawn (single, open)
  'T' = treasure (single, open)

Layout is rows × cols. (0,0) is top-left in game coordinates.
"""

# 15 cols × 9 rows.
# Two main vertical corridors at col 1 (left) and col 13 (right).
# Horizontal corridors at rows 2, 6 connect them fully.
# Row 4 has partial corridors on each side (detours but no central crossing).
# This biases all routes toward the screen edges where the YOLO detector works.
DEFAULT = [
    [1, 1, 1, 1, 1, 1, 1, 'T', 1, 1, 1, 1, 1, 1, 1],   # row 0
    [1, 0, 1, 1, 1, 1, 1,  0,  1, 1, 1, 1, 1, 0, 1],   # row 1
    [1, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 1],   # row 2 — top corridor
    [1, 0, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 0, 1],   # row 3
    [1, 0, 0, 0, 0, 0, 1,  1,  1, 0, 0, 0, 0, 0, 1],   # row 4 — partial mid corridor
    [1, 0, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 0, 1],   # row 5
    [1, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 1],   # row 6 — lower corridor
    [1, 0, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 0, 1],   # row 7
    [1, 0, 0, 0, 0, 0, 0, 'S', 0, 0, 0, 0, 0, 0, 1],   # row 8 — spawn row
]
