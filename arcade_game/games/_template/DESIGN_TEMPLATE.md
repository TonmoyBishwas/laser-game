# <Your game> — design notes (template)

Copy this file into your new game folder and fill it in. A short design note
keeps you (and your AI assistant) honest about staying laser-feasible.

## One-line idea

> e.g. "Slice the falling fruit with the pointer; avoid bombs."

## The core input verb

Which of these is the game built on? (See `docs/INPUT_CONSTRAINTS.md`.)

- [ ] **Point** — just where the pointer is (painting, guiding, paddle)
- [ ] **Dwell** — hold on a target to act (selecting, placing, catching)
- [ ] **Swipe** — move fast through a target (slicing, popping)
- [ ] **Trace** — draw a path (mazes, shapes)

## Targets and forgiveness

- Smallest thing the player must hit: ____ px (aim for ≥ ~80–100 px).
- What happens on a brief pointer dropout? (Should be gentle — no hard reset.)

## Win / lose / scoring

> e.g. "Score per fruit; 3 misses = game over; bombs end instantly."

## Art list (green-free!)

> e.g. fruit_*.png, bomb.png, background.png, slice.ogg — see `docs/ASSET_GUIDE.md`.

## Feasibility check

Could one slightly-shaky laser dot, that can't click and sometimes blinks off,
play this? If any part is "no", redesign it around dwell/swipe/trace before
building.
