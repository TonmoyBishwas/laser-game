# Background Diversity Pack — Green-Laser Data Collection

**Goal:** project a wide, deliberately varied set of **non-green** backgrounds, sweep
the green laser over all of them, and record. This trains a *background-invariant*
green-dot detector that works on the current maze **and** every future game
(fruit-ninja-style, etc.) — without needing the final art to exist yet.

**Why this works (the principle):** in object-detection research this is called
*domain randomization*. The finding that matters for us: **diversity beats realism** —
"increasing dataset diversity with complex/varied backgrounds" is what closes the gap,
and "photo-realism is not as important as diversity." If you randomize the background
hard enough, **the only constant the model can latch onto is the green dot itself.**
That's a detector that generalizes to art it has never seen.

---

## The one hard rule: NO bright/saturated GREEN in any background

Green = laser. Keep it that way.
- **Exclude** any image with large bright/saturated green areas (lush grass fields,
  neon-green UI, green-screen, etc.).
- A little **muted/dark** green (a forest at dusk, a houseplant in the corner) is
  tolerable and even useful as a mild distractor — but when in doubt, skip it.
- Most stock sites let you **filter/search by color** — use it to avoid green.

---

## The pack — 8 categories (~60–80 images total)

Aim for a slideshow of **60–80 images**, auto-advancing every **3–4 s**, on loop.
Combined with continuously moving the laser, that's enormous diversity in 25 minutes.

| # | Category | ~Count | Why it's in the pack |
|---|---|---|---|
| 1 | **Your real game art** (current maze + any concept art the artist has) | 5–10 | The actual deployment look — anchors the model to your real content. |
| 2 | **Colorful 2D game backgrounds** (platformer, arcade, fantasy, space, underwater) | 12–15 | Matches your planned future games (fruit ninja, etc.). |
| 3 | **Solid colors + gradients + color-bar test cards** (non-green) | 10 | Tests the dot on every uniform color — red, blue, white, black, yellow, purple. The cleanest contrast stress-test. |
| 4 | **Bright / high-key scenes** (sunny, neon, white, snow) | 8–10 | The HARD case: a green dot on a bright background has low contrast and the white bloom blends in. You *need* these. |
| 5 | **Dark / low-key scenes** (night, space, dungeon) | 8 | Easy case, but realistic — your maze is dark. The dot is super-bright here. |
| 6 | **Busy / textured / cluttered** (mosaics, patterns, foliage, markets, abstract) | 8–10 | Tons of edges and distractors → forces the model to ignore the background. |
| 7 | **Real-world photos** (landscapes, interiors, objects, food, streets) | 10 | Broad domain randomization — variety the model would never otherwise see. |
| 8 | **Arcade / laser-game aesthetics** (fruit-ninja fruit, shooting gallery, whack-a-mole, neon) | 6–8 | Directly previews the visual style of the games you plan to build. |

---

## Where to get them (free, no-attribution, current)

**Photos & gradients (categories 3,4,5,6,7):**
- **Pixabay** — https://pixabay.com — 6M+ images, **has a color filter** (great for avoiding green). Search: `colorful abstract`, `neon`, `gradient`, `night city`, `texture pattern`, `bokeh`.
- **Unsplash** — https://unsplash.com — 3M+ high-res, CC0. Search: `colorful`, `abstract background`, `neon lights`, `dark moody`.
- **Pexels** — https://pexels.com — millions, no attribution. Search: `vibrant background`, `bright`, `pattern`.
- **Color/gradient/test cards:** Pixabay search `gradient` / `color palette`; or just search `SMPTE color bars` and `solid color wallpaper red/blue/yellow/purple`.

**Game art (categories 2,8):**
- **Kenney** — https://kenney.nl (and https://kenney.itch.io/kenney-game-assets) — 40,000+ CC0 assets, clean & colorful, no sign-up. Grab the 2D background/space/platformer packs.
- **OpenGameArt** — https://opengameart.org — decade-deep library of 2D backgrounds (filter CC0).
- **itch.io** — https://itch.io/game-assets/free — huge "free game assets" section; search `2D background pack`, `parallax background`.

**Fruit-ninja-style reference (category 8):** on any of the above, search `fruit splash`, `cartoon fruit`, `arcade neon`, `shooting gallery`.

---

## How to assemble & project it (5 minutes)

0. **Generate the built-in part of the pack first** (both write into `bg_pack/`,
   which is gitignored — regenerate on any machine after `git pull`):
   - `venv\Scripts\python.exe make_bg_pack.py` — ~58 green-safe solids,
     gradients, textures, bright/dark scenes and white-glint negatives.
   - `venv\Scripts\python.exe render_game_frames.py` — ~42 REAL rendered frames
     of the menu, the thief maze (spotlight in many positions, ESCAPE banner)
     and fruit ninja. These are category 1 (your real game art) done for you.
1. Download extra photos/game art into the same `bg_pack/` folder, **delete any green-dominant images**.
2. Rename so they shuffle in a varied order (or rely on category mixing).
3. Project fullscreen, auto-advancing, looped. Easiest options:
   - **PowerPoint:** Insert → Photo Album → all images; Transitions → auto-advance 3–4 s; Slide Show → Set Up → loop continuously.
   - **Windows Photos / IrfanView:** open the folder → Slideshow → set 3–4 s interval, loop.
   - Or make it a video (any slideshow tool) and loop it.

Project this on the **real screen**, film with the **real phone camera** at the **real
tripod angle**, under **uni-bright lighting** (record at **1080p / 30fps**).

---

## Integrate with your 40-minute collection plan

While the slideshow auto-advances, keep the laser **moving continuously** and cover:

- **0–5 min** — set up: tripod where the game camera sits, slideshow projecting, phone recording 1080p.
- **5–28 min — record (laser ON), sweeping the dot over the cycling backgrounds:**
  - Hit **every screen position, center-heavy** (fixes the old center weakness).
  - Deliberately pause the dot on **bright/high-key slides** (cat. 4) — the hard case.
  - Vary **distance** (big near dot ↔ tiny far dot) and **angle** (round ↔ elliptical).
  - Mix **motion**: slow drags, fast flicks (motion blur), dead-still holds.
- **28–35 min — NEGATIVES (laser OFF):** let the same slideshow run with **no laser**,
  and add bright **white highlights / glints / light objects** in frame. ~15–20% of
  total footage. Teaches "a bright spot alone is not a laser" (your false-positive
  risk, since the dot blooms white-cored).
- **35–40 min** — buffer / re-shoot anything shaky / start the Roboflow upload.

**Target:** ~**400–800 labeled images** after Roboflow samples ~1–2 frames/sec.

---

## Roboflow settings (recap)

- **Sample 1–2 frames/sec** (avoid near-duplicate adjacent frames).
- **Preprocess:** auto-orient + resize to 640.
- **Augment (mild):** brightness/exposure + small blur/rotation. **Do NOT** apply heavy
  hue/saturation shifts — green is the signal, don't recolor it.
- **One class: `laser`.** Tight, consistent box on the bright core.
- Keep ~15–20% **null/background (no-laser)** images.

---

### Sources
- Pixabay: https://pixabay.com — Unsplash: https://unsplash.com — Pexels: https://pexels.com
- Kenney: https://kenney.nl — OpenGameArt: https://opengameart.org — itch.io: https://itch.io/game-assets/free
- Domain randomization / diversity-over-realism: arXiv 2506.07539, arXiv 2405.19822
