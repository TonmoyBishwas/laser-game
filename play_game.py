"""
Run a game with the MOUSE — no camera, laser, or projector needed.

This is the launcher for building and play-testing games. The mouse stands in
for the laser: move it to aim the "laser", move it off the window to make the
beam go dark. If a game plays well here with the mouse, it will play well with
the real laser.

    python play_game.py                       # menu (pick a game with the pointer)
    python play_game.py --game thief          # launch a game directly (fullscreen)
    python play_game.py --game thief --windowed   # resizable window (best for dev)
    python play_game.py --game thief --hold   # pointer only emits while holding the mouse button
    python play_game.py --list                # list games

To run the SAME game with the real laser + camera + projector instead:

    python play_laser_arcade.py --profile uni --device 0 --fullscreen
"""

from __future__ import annotations

import os
import sys
import argparse

_REPO = os.path.dirname(os.path.abspath(__file__))
# The game engine + games live under arcade_game/. Insert at the FRONT so
# `import arcade` finds this local package.
sys.path.insert(0, os.path.join(_REPO, "arcade_game"))


# ── Game registry ────────────────────────────────────────────────────────────
# name -> factory(app) -> Scene.  Imports are lazy so launching one game does not
# import the others. Add a new game by adding one entry here.
def _thief(app):
    from games.thief.game import create
    return create(app)


def _fruit_ninja(app):
    from games.fruit_ninja.game import create
    return create(app)


GAMES = {
    "thief": _thief,
    "fruit_ninja": _fruit_ninja,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a laser-arcade game with the mouse")
    parser.add_argument("--game", choices=list(GAMES), help="launch a game directly")
    parser.add_argument("--windowed", action="store_true", help="resizable window instead of fullscreen")
    parser.add_argument("--hold", action="store_true",
                        help="pointer only active while the left mouse button is held")
    parser.add_argument("--list", action="store_true", help="list games and exit")
    args = parser.parse_args()

    if args.list:
        print("Available games:")
        for name in GAMES:
            print(f"  {name}")
        return

    # Import the engine only after --list so a quick listing needs no display.
    from arcade.scene import GameApp, MenuScene

    app = GameApp(GAMES, windowed=args.windowed, hold_to_emit=args.hold)
    start = GAMES[args.game](app) if args.game else MenuScene(app)
    app.run(start)


if __name__ == "__main__":
    main()
