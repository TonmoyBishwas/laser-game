# Laser Guard — Paper-Writing Handoff

> **You are a fresh Claude Code instance.** Your job: help the user write an
> academic paper (university ML course final project) about the "Laser Guard"
> project in this repo. You have no prior memory of this project — **this
> document is your complete context.** It covers the goal, the full codebase,
> the methods, the entire build/debug journey, results, current state, and next
> goals, plus a suggested paper outline, a results table, and a draft abstract.
>
> **Academic-integrity rule (important):** numbers below are tagged
> **[MEASURED]** (came from training output, args.yaml, or our run logs — safe
> to cite) or **[ESTIMATE]** (approximate / not freshly re-verified — re-measure
> before putting in the paper). See §12 for the list of things to verify.
>
> Frame the paper as a **balanced methods + system** contribution: the detection
> method, the RL method, the integrated system, AND the deployment engineering
> all matter roughly equally.

---

## 1. How to use this document

- Read it top to bottom once; it is self-contained.
- §2–§3 give you the elevator pitch and architecture.
- §4 documents every source file (so you can describe the implementation).
- §5 is the methods you'll write up.
- §6 is the engineering journey (great material for a "Challenges/Deployment"
  section — the user explicitly wants this included).
- §7 is the results table. §10–§11 are a ready outline + draft abstract.
- §12 lists numbers to re-verify before citing.

---

## 2. Project goal & overview

**Laser Guard** is an interactive projector game that fuses computer vision and
reinforcement learning into a physical, real-time experience:

- A **projector** displays the game on a wall/screen.
- A **phone camera on a tripod** films the projected surface and streams to a PC.
- The player shines a **red laser pointer** on the projection. A **fine-tuned
  YOLO detector** locates the laser dot in each camera frame.
- A **homography** maps the laser's camera-pixel position into game coordinates.
- In the game, almost everything is dark; the laser acts as a **flashlight** that
  reveals a small lit region. Hidden in the dark is a **thief** moving through a
  **maze** toward a **treasure**.
- The thief is driven by a **reinforcement-learning policy** (tabular
  Q-learning). The player must hold the flashlight on the thief for ~1 second to
  **catch** it; the thief wins if it reaches the treasure.

**Academic context:** final project for a university ML course. The deliverables
are a working demo and this paper. The novelty is the **end-to-end integration**:
a custom-trained detector + an RL adversary + a real-time projected game, made to
run on commodity hardware.

**Two-environment reality (a recurring theme in the journey):**
- **Home / dev:** desktop with an **NVIDIA RTX 3090** (CUDA), fast.
- **Uni / demo:** a friend's **Intel Core i7 (12th gen), 16 GB RAM, no discrete
  GPU**, but with **Intel Iris Xe integrated graphics**. Everything ultimately
  has to run smoothly here for the live demo.

---

## 3. System architecture

Real-time, multithreaded, decoupled so slow inference never blocks rendering:

```
 Phone camera ──(USB virtual webcam)──> PC
        │
   CaptureThread        reads newest frame only (buffer size 1), no queue
        │  SharedState.latest_frame + frame_event
        ▼
   DetectorThread       YOLO inference on the latest frame
        │               → pick highest-confidence box → box centre (cx,cy)
        │               → homography: camera px → game px (gx,gy)
        │  SharedState.laser_state = (pos, conf, timestamp, camera_pos)
        ▼
   Pygame main loop     60 fps target; reads latest laser_state;
   (game_loop.run)      draws maze, thief, dark overlay + spotlight hole,
                        treasure, catch ring, edge glow, HUD
```

- **Threads** communicate through a `SharedState` object (locks + an event).
- The detector publishes the **latest** laser position; the render loop always
  uses the freshest one — detection rate and render rate are independent.
- A device **profile** (`home` / `uni`) sets YOLO device/imgsz/conf + camera
  index before threads start.

Entry point: `python -m game.main --profile {home|uni} [--recalibrate] [--debug]`.

---

## 4. Codebase documentation

Repo root: `H:\Projects\ml-project` (git repo, GitHub:
`github.com/TonmoyBishwas/laser-game`). Python 3.11+; deps: torch, ultralytics,
pygame, opencv-python, numpy (+ onnxruntime / openvino for fast CPU/iGPU paths).

### 4.1 `game/` package (the actual game)

| File | Purpose & key contents |
|---|---|
| `__init__.py` | Empty package marker. |
| `config.py` | All constants. Display `GAME_W,GAME_H=1920,1080`, `TARGET_FPS=60`. Model: `MODEL_PATH` (→ best.pt), `YOLO_CONF=0.25`, `YOLO_IMGSZ=640`, `YOLO_DEVICE="cuda:0"`, `MAX_DETECT_FPS=0`. Spotlight: `SPOTLIGHT_RADIUS=160`, `SPOTLIGHT_FADE_SECS=0.3`, `LASER_STALE_SECS=0.10`. Maze: `GRID_COLS,GRID_ROWS=15,9`, `CELL_W,CELL_H=128,120`, `CELL_TRANSITION_SECS=1.20`. Catch: `CATCH_HOLD_SECS=1.0`, `CATCH_RADIUS=100`, `CATCH_DECAY_RATIO=0.5`. Edge glow: depth 60, base 0.45, pulse 0.55. `ROUND_TIMEOUT_SECS=90`. |
| `main.py` | Entry point. `PROFILES` dict (home/uni — see §5.5). Applies profile to `config` before threads import it. Builds `SharedState`, opens camera, runs/loads calibration, starts `CaptureThread` + `DetectorThread` (+ optional debug preview thread), runs `game_loop.run`. CLI: `--profile --recalibrate --device --fullscreen --debug --policy`. |
| `calibration.py` | `Calibration` class: 4-corner click → `cv2.findHomography` → 3×3 matrix; `transform(cx,cy)` via `cv2.perspectiveTransform`; `save/load` JSON; `load_or_run(cap, force)`. Click order TL→TR→BR→BL; game corners are the 1920×1080 rectangle. |
| `capture.py` | `CaptureThread`: opens camera with `cv2.VideoCapture(index, cv2.CAP_ANY)`, buffer size 1; loops reading the newest frame into `SharedState`; retries on failure. |
| `detector.py` | `DetectorThread`: loads YOLO (`.pt`/`.onnx`/openvino dir), runs `predict(conf,imgsz,device)`, picks highest-conf box, homography-transforms its centre, publishes `LaserState`. `DetectorStats` (infer/detect fps, totals). `_resolve_device` falls back cuda→cpu and intel:gpu→intel:cpu. Honors `MAX_DETECT_FPS` (sleeps to cap rate). `.to()` only called for `.pt`. |
| `maze.py` | `Maze(grid)`: `is_wall`, `neighbours`, `step(r,c,action)`, `cell_center`, `pixel_to_cell`, `signature` (hash to match a policy to its maze). Actions N/E/S/W = 0/1/2/3. Grid encoding 0=path,1=wall,'S'=spawn,'T'=treasure. |
| `maze_layouts.py` | `DEFAULT` 15×9 maze: vertical corridors near col 1 and col 13, horizontal connectors; **edge-biased** so routes hug the screen edges where detection is best. Spawn (8,7), treasure (0,7). |
| `thief.py` | `Thief(maze, policy)`: at each cell centre queries `policy.act(state, greedy=True)`, animates a slide to the next cell over `CELL_TRANSITION_SECS`. `catch_progress` fills while laser within `CATCH_RADIUS`, decays at half speed otherwise; `caught` at ≥1.0. `reached` when at treasure. Draw methods incl. silhouette (debug) and catch ring. |
| `rl_env.py` | `ThiefEnv` (gym-style) + `LaserSim`. State `(thief_r,thief_c,laser_r,laser_c)`; `(-1,-1)`=no laser. Rewards (see §5.3). `LaserSim` modes: idle / patrol / tracking(lag+jitter) / random. `DT_SIM=0.4`, `MAX_STEPS=250`, catch needs ~3 consecutive on-thief steps (≈1.0 s). |
| `rl_policy.py` | `QPolicy`: sparse dict Q-table; `act(state,greedy)`, `update(...)` (Q-learning Bellman), `save/load` to compressed `.npz` with metadata (episodes, α, γ, ε schedule, maze signature, final win/caught rate, trained_at). |
| `game_loop.py` | Pygame render loop. Colors incl. `BG_COLOR=(8,8,18)`, walls cyan, treasure gold, `OVERLAY_ALPHA_MAX=245`, edge glow red. Render order: maze surface → thief → dark overlay with punched spotlight hole → treasure (always lit) → catch ring → edge glow → HUD (+ debug HUD: detector stats, laser coords, Q-value bars, thief silhouette). Spotlight fades in/out via `spot_alpha`; laser considered fresh within `LASER_STALE_SECS`. |

### 4.2 Top-level tools (built during development; not the game itself)

| File | Purpose |
|---|---|
| `train_thief.py` | Trains the Q-policy. Defaults: 20k episodes, α0.1, γ0.95, ε 1.0→0.05 over 80%, logs every 500. Saves `policies/thief_qtable.npz`. ~5 s on the 3090. |
| `find_cam.py` | Enumerates camera indices × backends (AUTO/DSHOW/MSMF), reports opened/frame/mean-brightness, previews each so you can identify the right camera. Disables MSMF HW transforms. `--fourcc/--width/--height` to fight green feeds. |
| `cam_diag.py` | Deep camera diagnostic: reads 60 frames, reports whether brightness CHANGES (live) or is FROZEN (constant), prints negotiated FOURCC/convert_rgb, saves sample PNGs. This proved DroidCam's feed was a frozen buffer. |
| `record_color_clips.py` | Cycles 10 solid-colour fullscreens, records the camera per colour to `clips/NN_color.mp4` for offline detection analysis. |
| `make_bg_images.py` | Writes 30 solid-colour 1920×1080 PNGs to `backgrounds/` for eyeball inspection. |
| `bg_palette.py` | Shared 30-colour palette (10 primaries/greys + 20 darker variants chosen for red-laser contrast). Used by the BG tools. |
| `live_bg_test.py` | Two modes. (a) Auto leaderboard: cycles the 30 colours, sweeps the laser, scores each by hits/sec, mean_conf, miss%, fp/sec, writes `bg_test_results/`. (b) `--hold`: lock one colour, arrow-key browse, live detection counters + a 30-cell green/red history strip + big indicator. Infinity-mirror-safe (never draws the camera feed). |
| `tune_detect.py` | Slider tuner for conf/imgsz/preproc, live or replaying a recorded clip; saves `detector_tuning.json`. |
| `eval_video.py` | Offline: runs YOLO over a clip/folder at chosen device/imgsz/conf, draws boxes, prints + saves per-clip hit%/conf/fps to `eval_results/`. Downscales preview for 4K. |
| `live_detect_cam.py` | Live webcam YOLO preview (boxes + fps + indicator) for a quick "does it detect?" check at a wall. `--device --yolo --model --imgsz --conf`. |
| `live_detect_rtsp.py` | Same but from an RTSP URL (reads `rtsp_url.py`). |
| `test_rtsp.py` | RTSP connectivity test: opens the URL, reports LIVE vs FROZEN. |
| `rtsp_url.py` | Single editable `RTSP_URL` constant shared by the RTSP tools. |
| `export_model.py` | Exports `best.pt` → ONNX or OpenVINO for faster CPU/iGPU inference. |
| `flashlight_sandbox.py` | **Key tuning tool.** The flashlight reveal in isolation (no thief / no game-over / infinite), using the real camera+YOLO+calibration pipeline. On-screen sliders (mouse or keyboard) for radius, offset x/y, smoothing, fade_secs, stale_secs, falloff, dark_alpha; raw (red +) vs corrected (green +) crosshairs; `--model --imgsz --yolo-device --max-detect-fps --render-fps`. Saves `flashlight_tuning.json`. |
| `run_best_flashlight.bat` | Launches the preserved known-good flashlight config (PyTorch .pt, imgsz 320, uncapped infer, render 30). |
| `setup_home.bat` / `setup_uni.bat` | One-time venv + deps install (CUDA torch vs CPU torch). |
| `ind_det.py`, `yyy.py`, `rtsp_detect.py` | **Experimental / scratch — not part of the system.** Ignore for the paper unless the user says otherwise. |

### 4.3 Documentation files

| File | Purpose |
|---|---|
| `README.md` | Project overview, quick-start, architecture, 30-second RL summary. |
| `README_UNI.md` | 6-phase uni-day demo checklist. |
| `RUN_COMMANDS.md` | Copy-paste command reference for all tools + flags. |
| `CAMERA_FIX.md` | DroidCam green-feed post-mortem + OBS workaround. |
| `MOBILE_CAM_APP_NOTES.md` | Spec for a future Android MJPEG-over-HTTP camera app to replace DroidCam. |
| `PAPER_HANDOFF.md` | This file. |

### 4.4 Model / data / results

- `runs/detect/laser_detector_run/weights/best.pt` — the trained detector (tracked in git). `best.onnx` also committed (exported).
- `runs/detect/laser_detector_run/args.yaml` — training config snapshot.
- Gitignored (regenerate locally): `Laser-Detection-1/` (Roboflow dataset),
  `clips/`, `backgrounds/`, `bg_test_results/`, `eval_results/`,
  `calibration.json`, `detector_tuning.json`, `policies/`, `venv/`.

---

## 5. Methods (for the paper)

### 5.1 Laser detection (computer vision)
- **Model:** YOLOv26-S (the "small" variant of YOLO v26), fine-tuned for a
  single class: the laser dot. **[MEASURED]** training config (args.yaml):
  base `yolo26s.pt`, **100 epochs**, **imgsz 640**, **batch 32**, device GPU 0,
  AMP on, patience 20, standard augmentation (HSV, scale 0.5, fliplr 0.5,
  erasing 0.4).
- **Dataset:** `Laser-Detection-1`, exported from Roboflow in YOLO format
  (train/val/test). Predominantly projector-screen images with a smaller share
  of monitor-screen images. **[ESTIMATE]** total image count (a few hundred) —
  verify.
- **Accuracy:** **[MEASURED]** **mAP50 ≈ 88.2%**.
- **Inference:** per frame, run YOLO, take the highest-confidence detection, use
  its bounding-box centre as the laser position.

### 5.2 Calibration (camera → game mapping)
- One-time **4-corner perspective calibration**: the operator clicks the four
  corners of the projected area in the camera image (TL→TR→BR→BL); the four
  game-space corners are the 1920×1080 rectangle. `cv2.findHomography` yields a
  3×3 matrix `H`; each detected laser point is mapped with
  `cv2.perspectiveTransform`. Saved to `calibration.json`.

### 5.3 RL thief (reinforcement learning)
- **Algorithm:** tabular **Q-learning** (sparse dict Q-table).
- **State:** `(thief_row, thief_col, laser_row, laser_col)`; laser `(-1,-1)` when
  off. ~15·9·(15·9+1) ≈ 18k nominal states (sparse in practice).
- **Actions:** 4 — N/E/S/W. Wall moves are no-ops with a penalty.
- **Rewards [MEASURED from rl_env.py]:** step into open cell **−0.01**; hit wall
  **−0.05**; caught by laser **−1.0** (terminal); reach treasure **+1.0**
  (terminal); timeout at 250 steps **−0.5** (terminal).
- **Training [MEASURED from train_thief.py defaults]:** 20,000 episodes, α=0.1,
  γ=0.95, ε-greedy decaying 1.0→0.05 over 80% of training, `DT_SIM=0.4 s/step`,
  catch requires ~1.0 s (≈3 steps) of laser-on-thief.
- **Robustness via scripted laser opponents:** episodes sample 4 laser modes —
  idle (learn navigation), patrol (lane avoidance), tracking-with-lag (human-like
  evasion), random.
- **Result:** **[ESTIMATE]** ~95% win rate (thief reaching treasure) against the
  scripted laser after training — re-verify from a fresh `train_thief.py` run,
  whose output prints final win/caught rates and stores them in the npz metadata.
- **Runtime use:** policy loaded greedily (ε=0); no learning during play.

### 5.4 Game mechanics
- **Flashlight reveal:** the screen is darkened by a near-opaque overlay
  (`OVERLAY_ALPHA_MAX=245`); a soft circular **hole** is punched at the laser
  position (radius `SPOTLIGHT_RADIUS`, soft falloff), revealing maze + thief only
  there. The treasure and pulsing **edge-glow direction hint** (stronger on the
  side the thief is nearer) are drawn above the overlay so they're always visible.
- **Hold-to-catch:** a progress ring fills while the laser stays within
  `CATCH_RADIUS` of the thief (~1 s to catch), decaying at half speed otherwise.
- **Edge-biased maze design:** because detection was weakest at screen centre and
  strongest near the edges (training-data bias), the maze deliberately routes the
  thief along the left/right gutters and edge corridors — a *design response to a
  model limitation* (good Methods/Discussion material).

### 5.5 Deployment profiles
**[MEASURED from main.py PROFILES]:**
- `home`: device `cuda:0`, imgsz 640, conf 0.25, camera index 0.
- `uni`: device `cpu`, imgsz 512, conf 0.20, camera index 2 (IriunCam).
The uni profile is the one that runs the live demo.

---

## 6. Engineering journey / struggles & iterations

This is the "how it really went" section (the user wants it in the paper).
Each item: **problem → diagnosis → fix**.

1. **Center-vs-edge detection weakness.** Detection was reliable at the left/right
   of the frame, poor in the centre. *Diagnosis:* training data was ~85%
   projector / ~15% monitor and under-represented centre conditions. *Fix:*
   redesigned the maze to be **edge-biased** so gameplay lives where detection
   works (rather than only chasing more data).

2. **DroidCam camera pipeline (the biggest time sink).** Used DroidCam (phone as
   USB webcam) and hit a cascade:
   - DSHOW backend raised "unknown C++ exception / can't capture by index".
   - MSMF opened a different/black device.
   - `cv2.CAP_ANY` at index 0 worked **on the home PC** but not portably.
   - On the uni laptops, OpenCV picked the **wrong camera** (built-in webcam, and
     a frozen dim feed that was actually the **IR / Windows-Hello camera**).
   - The DroidCam virtual webcam produced a **solid green / FROZEN feed**.
     *Diagnosis:* `cam_diag.py` showed a **constant mean brightness over 60
     frames** and an **empty FOURCC** → no real stream was negotiated; it was a
     static buffer. The MSMF-hardware-transforms env-var fix, forcing
     MJPG/YUY2/NV12, forcing resolution — none worked. Crucially, **OBS also got
     green** when reading the DroidCam *device*, while the DroidCam *client
     window* showed the live feed → the virtual-camera device itself was broken,
     machine-dependently.
   - *Workaround documented:* OBS **Window Capture** of the client window →
     OBS Virtual Camera (bypasses the broken device).
   - *Final decision:* **abandon DroidCam.** Switched to **IriunCam** (worked at
     camera index 2). Also tried **RTSP** from a phone app — worked but too laggy.
   - *Forward-looking fix:* spec'd a custom **Android MJPEG-over-HTTP** camera app
     (no Windows virtual-camera driver) — see `MOBILE_CAM_APP_NOTES.md`.

3. **Calibration through an "infinity mirror."** With the laptop mirrored to the
   projector, the calibration window (which shows the camera feed) recursed.
   *Fix:* click the **outermost** projected rectangle's corners; built
   infinity-mirror-safe tools that never display the camera feed
   (`live_bg_test.py --hold`, the game itself).

4. **Flashlight feel: offset + flicker.** Built `flashlight_sandbox.py` to tune
   the spotlight in isolation. Two findings: (a) the spotlight was **offset** from
   the laser — the game draws directly at the homography point with no
   correction, so the offset = **calibration imprecision** (worsened by the
   infinity-mirror calibration) and/or lens distortion; added live **offset x/y**
   correction + raw-vs-corrected crosshairs. (b) **Flicker** at low fps —
   `LASER_STALE_SECS=0.10` was shorter than the gap between detections (~104 ms at
   ~10 fps), so the light went stale between detections; exposed it as a slider
   (raise to ~0.3).

5. **CPU inference bottleneck (the performance arc — strong Results material).**
   On the uni i7 (no discrete GPU), making the flashlight feel *instant* was the
   hard part:
   - **PyTorch-CPU** inference at imgsz 512 was slow; at imgsz 320 ~**9–10 fps**
     and felt best of the CPU options. **[MEASURED]**
   - The **pygame fullscreen compositing competed with YOLO for CPU.** A naive
     sandbox dropped inference to **~3 fps**. *Fix:* reuse a single overlay
     surface (no per-frame 8 MB alloc), cache the scaled background, cap the
     render loop at 30 fps → frees CPU for inference. **[MEASURED]**
   - **ONNX Runtime** (Intel CPU) inference was **faster (~14–17 fps)** but then
     ran flat-out and **starved the render thread to ~1 fps** — the bottleneck
     flipped. *Fix:* `MAX_DETECT_FPS` cap so inference leaves CPU gaps for
     rendering. **[MEASURED]**
   - **OpenVINO on the Intel Iris Xe iGPU** (`predict(device="intel:gpu")`):
     **near-instant**, the best of all — because inference is **offloaded off the
     CPU**, removing the inference-vs-render contention entirely. **[MEASURED,
     qualitative]**
   - **Thermal throttling:** after ~1 min the iGPU drifts back toward baseline.
     *Diagnosis:* on Intel laptops the CPU and iGPU **share one power/thermal
     budget**, so sustained load throttles both. *Mitigations:* cap detection fps
     (less heat, still feels instant), plug in + best-performance power mode,
     laptop cooler.

6. **Preserved baseline.** Before experimenting further, the best PyTorch config
   was frozen as `run_best_flashlight.bat` + git tag `best-flashlight-baseline`.

---

## 7. Results / metrics (paper-ready)

| Metric | Value | Tag |
|---|---|---|
| Detection mAP50 | 88.2% | [MEASURED] |
| Detector | YOLOv26-S, 1 class, 100 epochs, imgsz 640, batch 32 | [MEASURED] |
| RL thief win rate vs scripted laser | ~95% | [ESTIMATE — re-verify] |
| RL training | 20k episodes, α0.1 γ0.95, ε1.0→0.05/80%, ~5 s on RTX 3090 | [MEASURED] |
| PyTorch-CPU inference @imgsz 320 (i7) | ~9–10 fps | [MEASURED] |
| Naive fullscreen sandbox (render starving YOLO) | ~3 fps | [MEASURED] |
| ONNX-Runtime CPU @320 (i7) | ~14–17 fps (but starved rendering until capped) | [MEASURED] |
| OpenVINO @ Iris Xe iGPU @320 | near-instant; smooth (CPU freed) | [MEASURED, qualitative] |
| Game render target | 60 fps (capped to 30 in tuning to free CPU) | [MEASURED] |
| Display / maze | 1920×1080; 15×9 grid; cells 128×120 px | [MEASURED] |

**Headline finding for the paper:** on commodity CPU-only hardware, *raw
inference speed was not the binding constraint — CPU contention between inference
and rendering was.* Offloading inference to the integrated GPU via OpenVINO
resolved it and was the single biggest real-time-performance win.

---

## 8. Current state

- End-to-end pipeline works: IriunCam → YOLO → homography → flashlight game.
- Detector supports `.pt`, ONNX, and OpenVINO; device fallbacks in place.
- **Chosen inference path for the demo: OpenVINO on the Iris Xe iGPU.**
- Flashlight tuned in the sandbox (radius/offset/stale/etc.); the PyTorch `.pt`
  baseline is preserved (launcher + git tag).
- The **OpenVINO + render-optimization recipe has been proven in the sandbox but
  not yet ported into the actual game** (`game/game_loop.py` still uses the older
  per-frame allocations; the `uni` profile still points at the `.pt`).

---

## 9. Next goals (post-holiday)

1. **Game integration:** point the `uni` profile at the OpenVINO model +
   `YOLO_DEVICE="intel:gpu"` (+ a sensible `MAX_DETECT_FPS` for thermals), and
   port the sandbox render optimizations (reused overlay surface, cached
   background, render-fps cap) into `game/game_loop.py`.
2. **Retrain** the detector on uni-condition footage (the recorded clips) to fix
   centre-screen and background-colour weaknesses; consider a nano model for CPU.
3. **Custom Android camera app** (MJPEG-over-HTTP, USB via adb-forward) per
   `MOBILE_CAM_APP_NOTES.md`, to retire DroidCam's fragility.
4. Optional: **INT8/OpenVINO quantization** for more iGPU speed; laptop cooler for
   sustained thermals.

---

## 10. Suggested paper outline

- **Abstract** — see §11.
- **1 Introduction** — the idea (laser-as-flashlight projector game), motivation
  (tangible CV+RL demo on commodity hardware), contributions (integrated system +
  deployment findings). Pull from §2.
- **2 Related Work** — single-class object detection / YOLO family; tabular
  Q-learning; interactive projection / AR games and laser-pointer interfaces.
  (The user/you should add proper citations.)
- **3 System Architecture** — the threaded pipeline. Pull from §3.
- **4 Methods** — 4.1 detection (§5.1), 4.2 calibration (§5.2), 4.3 RL thief
  (§5.3), 4.4 game mechanics incl. the edge-biased maze rationale (§5.4).
- **5 Implementation & Deployment Challenges** — the journey (§6). This is a
  differentiator; keep it analytical (problem→diagnosis→fix), not a diary.
- **6 Experiments & Results** — the table in §7; emphasize the inference-vs-render
  contention finding and the iGPU resolution.
- **7 Discussion / Limitations** — calibration sensitivity, thermal throttling,
  camera-pipeline fragility, dataset bias.
- **8 Conclusion & Future Work** — §9.

---

## 11. Draft abstract (refine before submission)

> We present **Laser Guard**, an interactive projector game that integrates a
> fine-tuned object detector and a reinforcement-learning agent into a real-time,
> physically-situated experience. A player aims a laser pointer at a projected
> scene; a YOLOv26-S model fine-tuned on a custom single-class dataset
> (mAP50 ≈ 88%) locates the laser dot in a phone-camera feed, and a homography
> maps it into game space, where it acts as a flashlight revealing an otherwise
> dark maze. A thief, controlled by a tabular Q-learning policy trained (20k
> episodes) against several scripted laser behaviours, navigates toward a
> treasure while the player tries to pin it with the light. Beyond the methods, we
> report the engineering required to run the system on commodity, GPU-less
> hardware, and our central deployment finding: real-time performance was limited
> not by inference speed but by **CPU contention between inference and rendering**;
> offloading inference to an integrated GPU via OpenVINO removed the bottleneck.
> We discuss calibration sensitivity, camera-pipeline fragility, thermal limits,
> and dataset bias, and outline a maze design that adapts gameplay to the
> detector's spatial reliability. *(~160 words — trim/adjust to the venue.)*

---

## 12. Numbers to VERIFY before citing

Re-measure these; do not cite the estimates as-is:
- **RL final win/caught rate** — re-run `python train_thief.py` (prints final
  rates) or read the metadata in `policies/thief_qtable.npz`. §5.3 says ~95%
  [ESTIMATE].
- **Dataset image count / split sizes** — check `Laser-Detection-1/data.yaml`
  and the train/val/test folders (gitignored; may need re-download).
- **mAP / precision / recall details** — 88.2% mAP50 is the figure on hand; for a
  fuller table re-run `yolo val model=runs/.../best.pt data=Laser-Detection-1/data.yaml`
  and report mAP50, mAP50-95, precision, recall.
- **Per-backend fps** — the §7 numbers are from live run logs on the specific uni
  i7; if the paper wants clean figures, re-run `eval_video.py` (offline,
  deterministic) at each device/imgsz, and/or the sandbox, and tabulate.
- **OpenVINO iGPU fps** — currently qualitative ("near-instant"); for a number,
  run `eval_video.py` is CPU-only, so instead read the sandbox's `infer=` log on
  `--yolo-device intel:gpu`.
- **Exact RL state-space / states-learned counts** — compute from the maze and
  the saved npz if the paper needs them.

When in doubt, prefer numbers you can regenerate from the repo over any figure in
this handoff.
