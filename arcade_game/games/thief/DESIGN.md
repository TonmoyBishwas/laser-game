# Thief game — design notes

The reference game. The pointer is a **flashlight** revealing a dark maze; a
thief sneaks from its spawn toward the treasure, and you catch it by holding the
light on it. This is the game that already runs on the real laser installation,
ported here to the mouse.

## How it plays

- The maze is dark. Wherever the pointer is, a circle of light reveals the walls
  and — if it's there — the thief.
- The thief moves cell to cell along the maze toward the treasure.
- **Catch it by keeping the light on it** for about a second. A ring fills around
  it; fill it completely to win (this is a *dwell*, the laser's way to "act").
- If the thief reaches the treasure, or time runs out, you lose.
- An edge glow hints which direction the thief is in.

## Files

| File | What it is |
|---|---|
| `game.py` | the Scene: rendering, the round, win/lose, flashlight overlay |
| `thief.py` | the thief: movement + the catch ring |
| `maze.py`, `maze_layouts.py` | the grid and the specific layout |
| `rl_policy.py`, `qtable.npz` | the thief's pre-trained brain (don't retrain here) |
| `constants.py` | the dials: spotlight size, catch time, thief speed, timing |

## What you can change freely

- **Colors / art**: restyle walls, thief, treasure, end screens. Swap the drawn
  rectangles for PNG sprites — load with `AssetLoader` in `game.py` and blit them
  centered on the entity position. (Keep it green-free.)
- **Feel**: spotlight radius and fade, catch time/radius, thief speed, round
  length — all in `constants.py`.
- **HUD / end screens**: wording, fonts, layout.

## What you must NOT change here

- **The maze walls or grid size.** The thief's brain (`qtable.npz`) is trained
  for this exact maze; changing the walls makes it wander. Re-skinning the maze
  (how walls look) is fine — only changing *which cells are walls* breaks it. A
  new maze is a request to the owner (it needs retraining elsewhere). See
  `CLAUDE.md` and `docs/PORTING_NOTES.md`.

## Turning the thief into a sprite (example)

In `thief.py`'s `draw`, instead of the rectangle:

```python
# once, in __init__: self.img = assets.load_image("thief.png", size=(THIEF_W, THIEF_H))
surface.blit(self.img, (int(self.pos[0] - THIEF_W/2), int(self.pos[1] - THIEF_H/2)))
```

(Pass an `AssetLoader` into the Thief from `game.py`, or load in `game.py` and
blit there.) Same approach for the treasure and walls.
