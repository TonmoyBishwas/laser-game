# Laser Guard — Uni Day Checklist

This is the **printable checklist** for tomorrow's test session and the
final demo day. Follow it top-to-bottom; nothing here should take more
than 15 min per phase.

---

## TONIGHT — prepare (10 min)

- [ ] Push latest code to GitHub (you're reading this, so the repo already has the bundle).
- [ ] Confirm your phone's DroidCam app is installed and recent.
- [ ] **Charge phone fully**, bring USB cable.
- [ ] Bring **laser pointer + spare** (yes, really).
- [ ] On your home PC, sanity-run: `venv\Scripts\python.exe -m game.main --profile uni`
      — this forces CPU + imgsz=320 so you preview the uni performance.
      Should still hit ~5-15 fps on a Ryzen 9 even on CPU.

---

## AT UNI — Phase 1: install + boot (15 min)

Friend's laptop, projector connected, phone on tripod facing projector.

1. **Clone the repo**
   ```
   git clone https://github.com/<you>/<repo>.git laser-game
   cd laser-game
   ```

2. **Run the one-click setup**
   ```
   setup_uni.bat
   ```
   This installs CPU torch (~200 MB) + everything else into `venv\`. Goes for ~5 min on uni Wi-Fi.

3. **Plug phone via USB, open DroidCam (phone + Windows app), click "USB"** in the PC app. Confirm you see the projector image in the DroidCam preview.

4. **Find the camera index**
   ```
   venv\Scripts\python.exe find_cam.py
   ```
   Note which index shows the DroidCam feed (usually 0 with backend AUTO). Use `--device N` later if it's not 0.

5. **First boot of the game, with calibration**
   ```
   venv\Scripts\python.exe -m game.main --profile uni --recalibrate --debug
   ```
   Click the 4 corners of the projected area in order: **TL → TR → BR → BL**, press Enter.

### Decision point #1
Aim the laser at the projector and watch the **debug HUD** + the **YOLO preview window**:

- Detection FPS ≥ 10 and boxes track the laser ✅ → skip Phase 2 colour tests, go to Phase 4
- Detection FPS 3-10, boxes track but slow ⚠️ → continue with Phase 2
- Detection FPS < 3 or no boxes ❌ → continue, but plan to retrain at home

---

## Phase 2 — record colour clips (10 min)

```
venv\Scripts\python.exe record_color_clips.py --duration 30
```

For each of the 10 colours (script handles the cycling), during the 30 s **do this script-religiously**:

| sec  | what |
|------|---|
| 0–5  | hold dead-centre |
| 5–10 | sweep through all 4 corners |
| 10–15| top edge, left-to-right |
| 15–20| bottom edge, left-to-right |
| 20–25| fast random motion |
| 25–30| 3 holds in different spots |

Output: `clips/01_black.mp4` … `clips/10_gray25.mp4` + `clips/metadata.json`.

Press `SPACE` to skip a colour if you already covered it; `ESC` aborts.

---

## Phase 3 — tune per colour (15 min)

For each clip:
```
venv\Scripts\python.exe tune_detect.py --source clips/01_black.mp4
... etc
```

In the tuner window, watch the `detections raw=X` counter and the box quality. Record on paper:

| colour | best conf | detect fps | centre OK? | notes |
|--------|-----------|-----------|------------|-------|
| black  |  .  |  /30 |  |  |
| white  |  .  |  /30 |  |  |
| red    |  .  |  /30 |  |  |
| green  |  .  |  /30 |  |  |
| blue   |  .  |  /30 |  |  |
| cyan   |  .  |  /30 |  |  |
| magenta|  .  |  /30 |  |  |
| yellow |  .  |  /30 |  |  |
| gray50 |  .  |  /30 |  |  |
| gray25 |  .  |  /30 |  |  |

Press `s` in the tuner on your best-performing colour to save it as `detector_tuning.json`.

**Top 2 colours by detection FPS = primary + backup game background.**

---

## Phase 4 — play the actual game on the winning colour (15 min)

1. Edit `game/game_loop.py` → change `BG_COLOR` to your top colour's RGB:
   ```python
   BG_COLOR = (R, G, B)
   ```
2. Run the game:
   ```
   venv\Scripts\python.exe -m game.main --profile uni --debug
   ```
3. Friend plays for 5 min. You watch:
   - Does the spotlight follow the laser smoothly?
   - Does the thief get caught reliably when laser overlaps it?
   - Does the thief reach the treasure within ~30 s of dodging?
4. **Record a 1-minute phone video** of the gameplay — presentation backup.

### Decision point #2

- Game feels fun ✅ → skip Phase 5, go to Phase 6
- Detection laggy ⚠️ → already on `--profile uni` (CPU + imgsz 320). Nothing more to do at uni — note it for home day 2.
- Detection misses obviously visible laser ❌ → continue to Phase 5

---

## Phase 5 — targeted bonus recordings (15-20 min, only if needed)

For each problem region/colour combo, repeat:
```
venv\Scripts\python.exe record_color_clips.py --duration 60 --outdir clips_extra
```
But during recording, **only aim at the problem area** (e.g. screen centre on white background). Hold long, slow, varied.

These clips are gold for retraining at home.

---

## Phase 6 — pack up (5 min)

- [ ] `git add clips/ detector_tuning.json calibration.json notes.md`
- [ ] `git commit -m "uni day clips + notes"`
- [ ] `git push`
- [ ] Photo the room setup (tripod position, distance, lighting).

---

## AT HOME — Day 2 (the day after uni)

Based on **Decision Point #1**:

### If detection was good
- Day 2: polish — sound, score, end screens, edge cases.
- 1-2 dry runs in `--profile uni` to confirm nothing regressed.

### If detection had specific gaps
1. **Extract frames** from problem clips for labelling:
   ```
   mkdir extra_frames
   ffmpeg -i clips/06_cyan.mp4 -vf fps=2 extra_frames/cyan_%04d.jpg
   ```
2. Upload to Roboflow, auto-label, fix any wrong boxes, export as YOLO format → `Laser-Detection-2/`.
3. **Resume training** on your 3090:
   ```
   yolo train resume model=runs/detect/laser_detector_run/weights/best.pt ^
     data=Laser-Detection-2/data.yaml ^
     epochs=30 ^
     imgsz=640
   ```
   ~30 min on RTX 3090.
4. The new `best.pt` lands in `runs/detect/laser_detector_run2/weights/`. Update `MODEL_PATH` in `game/config.py` or copy/replace.

---

## AT HOME — Day 3 (final dry run)

- Force the uni profile on your home PC: `--profile uni`.
- Run the entire demo flow end-to-end. Time it.
- If the game runs smoothly here on CPU at imgsz=320, **it will run on the uni i3 too** (your CPU is just faster but the ratio holds).

---

## Demo day (Day 4)

1. `git pull` on friend's laptop (already set up since uni Day 1).
2. Recalibrate against today's projector framing: `python -m game.main --profile uni --recalibrate`.
3. Friend's laptop on the lectern, your phone on the tripod facing the projector.
4. **You** are the player. Your friend's laptop just runs the game.

Good luck. If anything breaks last-minute, fall back to playing the phone video from Phase 4.
