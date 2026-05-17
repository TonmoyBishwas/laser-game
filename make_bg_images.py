"""
Generate 10 solid-colour PNG backgrounds at 1920x1080 into `backgrounds/`.

Open any of them fullscreen with your OS image viewer (Windows: F11 in Photos)
to inspect detection behaviour by eye without running any script.

  venv\\Scripts\\python.exe make_bg_images.py
"""

from pathlib import Path

import numpy as np
import cv2


COLORS = [
    ("01_black",   (  0,   0,   0)),
    ("02_white",   (255, 255, 255)),
    ("03_red",     (255,   0,   0)),
    ("04_green",   (  0, 255,   0)),
    ("05_blue",    (  0,   0, 255)),
    ("06_cyan",    (  0, 255, 255)),
    ("07_magenta", (255,   0, 255)),
    ("08_yellow",  (255, 255,   0)),
    ("09_gray50",  (128, 128, 128)),
    ("10_gray25",  ( 64,  64,  64)),
]

W, H = 1920, 1080


def main() -> None:
    outdir = Path("backgrounds")
    outdir.mkdir(parents=True, exist_ok=True)

    for name, rgb in COLORS:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # cv2 uses BGR, source tuples are RGB -> flip
        img[:] = (rgb[2], rgb[1], rgb[0])
        path = outdir / f"{name}.png"
        cv2.imwrite(str(path), img)
        print(f"wrote {path}")

    print()
    print("Done. Open any PNG fullscreen on the projector to test by eye.")
    print("Windows: right-click -> Open with -> Photos, then F11 for fullscreen.")


if __name__ == "__main__":
    main()
