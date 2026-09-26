"""
Render REAL frames of the arcade games (menu, thief, fruit ninja) headlessly and
save them into `bg_pack/` as `game_*.png` (1920x1080).

These are the exact screens the laser will be pointed at during play, so they
are the most valuable backgrounds in the dataset-recording slideshow. The games
are driven by a scripted pointer (a slow sweep) so the thief spotlight, fruit,
bombs, juice particles and score popups appear in natural positions.

  venv\\Scripts\\python.exe render_game_frames.py
  venv\\Scripts\\python.exe render_game_frames.py --thief 24 --fruit 24 --seed 3
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "arcade_game"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from arcade.config import GAME_H, GAME_W  # noqa: E402
from arcade.input import Pointer  # noqa: E402
from arcade.scene import GameApp, MenuScene  # noqa: E402
from games.fruit_ninja.game import create as create_fruit  # noqa: E402
from games.thief.game import create as create_thief  # noqa: E402

OUT = ROOT / "bg_pack"
DT = 1.0 / 30.0


class HeadlessApp(GameApp):
    """GameApp without a window: just a canvas + registry."""

    def __init__(self, registry):
        self.registry = registry
        self.scene = None
        self._running = False
        self.clock = pygame.time.Clock()
        pygame.display.set_mode((1, 1))
        self.canvas = pygame.Surface((GAME_W, GAME_H)).convert()

        class _D:  # minimal stand-in for arcade.display.Display
            pass
        self.display = _D()
        self.display.canvas = self.canvas


def sweep_pointer(t: float, prev: tuple[float, float] | None, active=True) -> Pointer:
    """Slow Lissajous sweep over the whole canvas (covers centre + edges)."""
    x = GAME_W / 2 + (GAME_W / 2 - 80) * math.sin(0.37 * t)
    y = GAME_H / 2 + (GAME_H / 2 - 80) * math.sin(0.61 * t + 1.1)
    vx, vy = (0.0, 0.0) if prev is None else (x - prev[0], y - prev[1])
    return Pointer(x, y, active, vx, vy)


def save(canvas: pygame.Surface, name: str) -> None:
    pygame.image.save(canvas, str(OUT / f"{name}.png"))
    print("wrote", name)


def greenish(canvas: pygame.Surface) -> float:
    """Fraction of pixels that are bright saturated green (should be ~0)."""
    a = pygame.surfarray.array3d(canvas)  # (W,H,3) RGB
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    m = (g > 140) & (g > r + 60) & (g > b + 60)
    return float(m.mean())


def render_menu(app: HeadlessApp) -> None:
    scene = MenuScene(app)
    scene.update(DT, Pointer(-1, -1, False))
    scene.draw(app.canvas)
    save(app.canvas, "game_menu_idle")
    # hovering a tile (dwell bar partially filled)
    tile = list(scene._tiles.values())[0].center
    for _ in range(12):
        scene.update(DT, Pointer(tile[0], tile[1], True))
    scene.draw(app.canvas)
    pygame.draw.circle(app.canvas, (255, 255, 255), tile, 10, 2)
    save(app.canvas, "game_menu_hover")


def render_game(app: HeadlessApp, factory, prefix: str, n_frames: int,
                total_secs: float, laser_off_every: int) -> None:
    scene = factory(app)
    steps = int(total_secs / DT)
    grab_at = sorted(random.sample(range(int(steps * 0.05), steps), n_frames))
    prev = None
    k = 0
    for i in range(steps):
        t = i * DT
        # every Nth snapshot is taken with the pointer OFF (laser not on screen)
        off = (k % laser_off_every == laser_off_every - 1) if laser_off_every else False
        p = sweep_pointer(t, prev, active=not off)
        prev = (p.x, p.y)
        scene.update(DT, p)
        if grab_at and i == grab_at[0]:
            grab_at.pop(0)
            scene.draw(app.canvas)
            if scene.show_pointer and p.active:
                pygame.draw.circle(app.canvas, (255, 255, 255), (int(p.x), int(p.y)), 10, 2)
            g = greenish(app.canvas)
            tag = "off" if off else "on"
            save(app.canvas, f"{prefix}_{k:02d}_{tag}")
            if g > 0.002:
                print(f"   WARNING {prefix}_{k:02d}: {g*100:.2f}% bright-green pixels")
            k += 1
        # restart the game if it ended so later frames still show play
        if getattr(scene, "game_over", False):
            if hasattr(scene, "reset"):
                scene.reset()
            elif hasattr(scene, "_reset"):
                scene._reset()
            elif hasattr(scene, "start_new_game"):
                scene.start_new_game()
            else:
                scene = factory(app)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thief", type=int, default=20, help="frames from the thief game")
    ap.add_argument("--fruit", type=int, default=20, help="frames from fruit ninja")
    ap.add_argument("--secs", type=float, default=120.0, help="simulated play seconds per game")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    OUT.mkdir(exist_ok=True)
    pygame.init()
    pygame.font.init()

    app = HeadlessApp({"thief": create_thief, "fruit_ninja": create_fruit})
    render_menu(app)
    render_game(app, create_thief, "game_thief", args.thief, args.secs, laser_off_every=4)
    render_game(app, create_fruit, "game_fruit", args.fruit, args.secs, laser_off_every=4)
    pygame.quit()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
