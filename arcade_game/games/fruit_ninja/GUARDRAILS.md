# Fruit Ninja — build guardrails

This folder is a **brief, not a built game**. Build the game here yourself (with
your AI assistant). This page tells you exactly how it must work so it plays the
same with the mouse now and with the **green laser** on the real installation
later. Follow it and the port is free; ignore it and the game won't survive the
laser.

Read `CLAUDE.md`, `docs/INPUT_CONSTRAINTS.md`, and `docs/ASSET_GUIDE.md` first.

---

## The core idea, in laser terms

- **The pointer is the blade.** Where the dot is, the blade is. There is no
  sword to swing and no button to press.
- **A "slice" is the pointer moving fast THROUGH a fruit.** Not a click, not a
  tap — the *path* of the dot crossing the fruit, while moving quickly. This is
  the single most important rule. A still pointer resting on a fruit must NOT
  slice it; only motion through it does.
- **One blade only.** There's one laser, so one cutting point. No multi-finger
  combos.
- **The blade can blink out.** If the pointer goes inactive (off-screen / dropout),
  the blade simply stops cutting that frame — don't crash, don't end a combo
  harshly on a one-frame gap.

## How to detect a slice (use the engine, don't reinvent it)

```python
from arcade import gestures

self.trail = gestures.Trail(max_points=14)     # in __init__
...
self.trail.update(pointer)                      # each frame in update()
seg = self.trail.last_segment()                 # (prev_pos, cur_pos) or None
if seg and gestures.is_fast(pointer, min_speed=25):
    p0, p1 = seg
    for fruit in self.fruits:
        if not fruit.sliced and gestures.segment_hits_circle(p0, p1, fruit.pos, fruit.radius):
            fruit.slice()
```

- `segment_hits_circle` checks the whole movement segment, so a fast swipe can't
  "tunnel" past a fruit between two frames.
- `is_fast` gates on speed so resting/aiming doesn't cut.
- Tune `min_speed` and `fruit.radius` for feel — err toward **generous** (big
  fruit, forgiving speed) because the laser is shakier than a mouse.

## What's allowed (do these)

- Fruit launched in arcs (gravity); slice them before they fall off-screen.
- **Combos**: multiple fruit cut in one continuous fast swipe → bonus.
- **Bombs / avoid-objects**: slicing one ends the game or costs a life — this is
  great, because "don't swipe through X" is fully expressible.
- A **blade trail** drawn behind the pointer for juice (`gestures.Trail.draw`,
  in a non-green color).
- Score, lives, rising difficulty (spawn rate / speed) over time.
- Sound on slice / bomb (optional; missing files are harmless).

## What's forbidden (never do these)

- ❌ Click / tap to slice. (No button exists.)
- ❌ Keyboard controls for the player. (Operator-only keys are fine: `R`/`M`/`Q`.)
- ❌ Two blades / multi-touch / co-op.
- ❌ Tiny fruit or pixel-precise hit zones. Make fruit big (~120 px).
- ❌ Punishing the player for a one-frame pointer dropout.
- ❌ **Green fruit or green UI.** Restyle any "naturally green" fruit to another
  color (blue melon, purple apple, orange lime). Green is the laser.

## Art (see ASSET_GUIDE for full rules)

Suggested pieces, dropped in `games/fruit_ninja/assets/` as transparent PNGs
(~240×240 export, shown ~120 px):

- `fruit_*.png` (a few non-green fruits), each ideally with a whole + two halves,
  or just a whole + a generic splat.
- `bomb.png`.
- optional `splat.png`, background, sounds (`slice.ogg`, `bomb.ogg`).

Until art exists, the loader shows labeled placeholders, so you can build and
play the whole game first, then drop in PNGs.

## Definition of done

- [ ] Fruit are sliced by **fast pointer movement through them**, never a click.
- [ ] Resting the pointer on a fruit does nothing.
- [ ] One blade; brief dropouts don't break it.
- [ ] Bombs punish slicing-through; everything is big and forgiving.
- [ ] No green anywhere.
- [ ] Registered in `run.py` and runs with `python run.py --game fruit_ninja --windowed`.
- [ ] You did not edit `arcade/`.
