# Running the game — NEW version vs. BACKUP

There are now **two** games in this repo. They share the **same** camera,
calibration, and laser-detection pipeline — only the game layer differs. Pick one
with the command you run.

## NEW game (A* thief, random mazes, escape doors, custom art)

Built and play-tested in the `laser-arcade` repo (with the mouse standing in for
the laser) and vendored here under `arcade_game/`. Driven by the real laser
through a `LaserPointer` adapter.

```powershell
# On the projector laptop (uni i7):
python play_laser_arcade.py --profile uni --fullscreen

# First time on a new setup, force calibration:
python play_laser_arcade.py --profile uni --fullscreen --recalibrate

# Camera index / phone-stream overrides work exactly like the old game:
python play_laser_arcade.py --profile uni --fullscreen --device 2
python play_laser_arcade.py --profile uni --fullscreen --url http://localhost:8090/video
```

## BACKUP game (original RL thief, fixed maze) — unchanged

If anything misbehaves with the new version, the original game still runs exactly
as before. Nothing about it was changed.

```powershell
python -m game.main --profile uni --fullscreen
```

A git tag **`backup-rl-game-2026-06-16`** also marks the exact state right before
this port, as an extra restore point.

## In-game keys (both versions)

- **R** — restart the round (after it ends)
- **Q** / **Esc** — quit

## How the new game is wired (for reference)

- `arcade_game/` — the vendored game engine + thief game (a 1:1 copy of the
  `laser-arcade` repo's `arcade/` + `games/thief/`). To pull in future changes,
  re-copy those folders over `arcade_game/`.
- `play_laser_arcade.py` — reuses `game/`'s camera → calibration → detector
  threads, then runs the vendored game with `LaserPointer` (real laser) instead
  of the mouse. The maps are 1920×1080 in both, so laser positions drop straight
  onto the game canvas with no extra conversion.
- `game/` — the original pipeline **and** the backup game. Left completely
  untouched by this port.
