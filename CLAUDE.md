# CLAUDE.md — rules for working in this repo (Laser Guard / LaserLearn)

**Read this before changing anything. If you are an AI coding assistant, treat
this file as binding instructions.** Most of these rules were paid for with a
broken teacher demo — the full story is in `DETECTION_POSTMORTEM.md`.

## What this repo is

The **deployment** repo: phone camera → `game/capture.py` (CaptureThread) →
`game/detector.py` (YOLO/OpenVINO, DetectorThread) → `shared.laser_state` →
a pygame game reads it as a pointer. Capture + detection run on **background
threads**; the game render loop is the **main thread**. They all share one CPU
under Python's **GIL** — that fact drives rule #1.

- Launchers: `python play_laser_arcade.py` (menu + thief + fruit ninja, the main
  game), `python -m game.main` (original RL game, backup),
  `python flashlight_sandbox.py` (isolated detector tuner — the known-good
  detection baseline).
- `arcade_game/` is a **vendored copy** of the separate `laser-arcade` repo
  (see "Vendoring" below).
- Demo machine: **uni i7** (12th gen, Intel Iris Xe iGPU, no NVIDIA, 16GB).
  Home dev PC: RTX 3090.

---

## Rule 1 — NEVER do per-frame work you could do once (the demo-killer)

The 2026-06 detection failure was NOT the camera, model, color gate, or
confidence. It was the game's main thread hogging the CPU/GIL with per-frame
pygame work, starving the detector thread until detection collapsed to zero.
The engine was fixed on 2026-07-03 (2.6–3.9× less main-thread work). Keep it
fixed:

- **No `pygame.Surface(...)` allocation inside `draw()`/`update()`.** Allocate
  scratch/overlay surfaces once in `__init__` and reuse (`fill()` to reset).
- **No `pygame.font.SysFont(...)` or `font.render(static text)` per frame.**
  Pre-render static text once; re-render dynamic text only when its value
  changes. (The old ESCAPE! banner built 2 fonts + 37 renders per frame.)
- **No `transform.smoothscale`/`rotate`/`flip` of the same image per frame.**
  Pre-compute (rotation buckets, pre-flipped frames, prescaled buffers).
- **Keep the detector throttled:** `MAX_DETECT_FPS: 12` in the `uni`/`uni-cpu`
  profiles (`game/main.py`). The GIL-releasing sleep in `game/detector.py` only
  fires if this is set. Never silently return it to 0/unlimited.
- **Keep render FPS capped at 30** for the laser build (the launcher's
  `--render-fps` default). 60 doubles the GIL pressure for no visible gain.
- Reference implementations of "done right": `flashlight_sandbox.py`,
  `game/game_loop.py`, and (since 2026-07-03) the arcade engine itself.

If a new game or effect needs something expensive every frame, budget it: the
whole main-thread frame should stay in the low single-digit milliseconds.

---

## Rule 2 — the debugging protocol when detection "gets worse"

Do these FIRST, in order, before touching any threshold:

1. **Run the decisive comparison:** `python flashlight_sandbox.py --device 0`.
   If the sandbox detects fine with the same camera + model, the fault is the
   **game process** (render load / GIL), not camera/model/gate/conf. This one
   test would have saved a full wasted day.
2. **Read the `--debug` stats line** (`[detector] infer=… detect=… gated=…`):
   - `infer` high, `detect` → 0, `gated` = 0 ⇒ detector is starved or the feed
     is dead — NOT a threshold problem.
   - `gated` climbing ⇒ the color gate is rejecting the real laser (exposure
     drift / washed-out dot) — fix the camera app (lock AE/WB/focus) before
     touching the gate.
3. **Change ONE variable at a time.** The failed debugging shipped many small
   speculative changes at once (gate tighten→loosen, conf 0.20→0.10, a camera
   watchdog for a freeze that wasn't happening) and muddied every signal.
4. Do not "fix" the detector's *output* (gate, conf, camera) when the symptom
   pattern says the detector is being *starved*.

Current band-aids that are deliberately kept (harmless, re-evaluate only after
steady detection is confirmed on the i7): the green color gate at H_LO=35, the
`uni` profile's conf 0.10, the camera freeze watchdog.

---

## Rule 3 — vendoring (`arcade_game/` ↔ the `laser-arcade` repo)

- **Engine + game logic is edited in `H:\Projects\laser-arcade` (source of
  truth), then re-vendored here.** Never hand-edit `arcade_game/` for
  game/engine logic. Pipeline files (`game/capture.py`, `game/detector.py`,
  `game/calibration.py`, `game/config.py`, `game/main.py`,
  `play_laser_arcade.py`) are edited directly in THIS repo.
- Re-vendor = robocopy `/MIR /XD __pycache__` per subdir (`arcade`,
  `games\thief`, `games\fruit_ninja`, `games\_template`). Then byte-verify and
  check `git status` shows only intended files.
- **Preserve `arcade_game/games/__init__.py`** — it exists only in this repo,
  not in laser-arcade; a careless full-tree mirror would delete it.
- Do NOT vendor laser-arcade's `games/fruit/` (legacy) or `games/tower_defense/`
  (uncommitted there).
- As of 2026-07-03 the two trees are fully converged (the old "thief fixes only
  exist in the vendored copy" warning in the postmortem is stale). Keep it that
  way: land changes in source first.

---

## Machines & profiles

- `--profile home` = CUDA + imgsz 640. **Use this at home.** Never run
  `uni-cpu` on the home PC (it ignores the 3090 and looks falsely slow).
- `--profile uni` = OpenVINO `best_openvino_model` on `intel:gpu`, imgsz 320,
  conf 0.10, MAX_DETECT_FPS 12. Primary for the uni i7.
- `--profile uni-cpu` = CPU `best.pt`, imgsz 512, MAX_DETECT_FPS 12. Fallback.
- Camera index varies per session (`--device 0/2`); `--url` takes an MJPEG
  stream (openwebcam). Keep the phone screen awake + camera app foregrounded.
- The i7 shares one power/thermal budget between CPU and iGPU — expect
  throttling on long runs; judge performance after a few minutes, not seconds.
- The user drives the uni laptop over Chrome Remote Desktop: **judge game speed
  by what's on the projector (friend's report), never by the remote-desktop
  stream** (it caps FPS and adds lag of its own).
- Validation checklist for any perf/detection change: `DETECTION_POSTMORTEM.md`
  §9 — `detect=` must hold ≥ ~12 fps and never collapse during real play, ON
  THE i7.

## Art / color

- **Bright or saturated green is reserved for the laser.** No green in game
  art, UI, or debug overlays — the detector will chase it. (The gold-treasure
  false detections were cured by hiding the treasure in the dark, not by
  tightening the gate.)

## Git & secrets (public repo!)

- **Never `git add -A` / `git add .`** — stage files explicitly.
  `train_green_yolo26n.ipynb` contains the Roboflow API key and must stay
  untracked. Same for stray notebooks/scripts at the repo root.
- Commit identity here: `TonmoyBishwas@users.noreply.github.com`; in
  laser-arcade: `ttonmoy46@gmail.com`. Both repos add the trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Never name the collaborator (no name/pronouns in files or commits).
- Push after every meaningful commit — the uni machine syncs via `git pull`.

## Model facts (so nobody re-derives them wrong)

- Deployed detector: **green-laser YOLO26-nano** (single class `laser`),
  dataset `green_train/Laser-Detection-Green-1/`, weights at
  `runs/detect/laser_detector_run/weights/` (`best.pt` + `best_openvino_model/`,
  both committed). Test mAP50 ≈ 0.881 @ 640, ≈ 0.816 @ 320 (deployed size).
- `runs/detect/laser_detector_run/args.yaml` is STALE — it describes the old
  red-laser training run, not the green model. Don't cite it.
- The old **red-laser YOLO26-S** lives at `runs/detect/laser_detector_run_red/weights/`
  (`best.pt` + `best_openvino_model/` @ 320, both committed). Select it with
  `--laser red` on `play_laser_arcade.py`, `game.main` and `flashlight_sandbox.py`:
  that swaps the model path AND the colour gate (`cfg.LASER_COLOR`, red hue bands
  wrap 0-10 / 170-179) together. Default stays green.
- The RL thief Q-table (`policies/thief_qtable.npz`) is tied to the fixed
  15×9 maze in the original game; the arcade thief uses A* and needs no
  training.
