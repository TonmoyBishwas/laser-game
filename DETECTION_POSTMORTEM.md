# Laser Guard — Detection Failure Post-Mortem (2026-06-16 → 2026-06-18)

> Durable record written before pausing the project ~1–2 weeks. When we resume, read this
> to recover full context: what broke, what was tried, the real root cause, and the fix.

## 0. TL;DR (read this first)

- **Symptom:** after the two new games (thief + fruit ninja) were vendored in, laser detection
  in the **full game** became inconsistent and eventually unplayable — the spotlight blinks,
  drops roughly every ~0.5s, occasionally jumps to a wrong spot, and you can't reliably select a
  menu tile or slice a fruit. A teacher demo on the **morning of 06-17 worked great**; by that
  **evening (same uni i7 laptop, after the day's changes) it was unplayable** — an embarrassing
  second demo.
- **Real root cause (confirmed by code investigation):** the **vendored `arcade_game/` engine
  does heavy per-frame work on the main thread** — above all a **full-screen
  `pygame.transform.smoothscale` every frame** (`arcade_game/arcade/display.py` `present()`),
  plus per-frame surface allocations, a **60 FPS** target, and an **unthrottled** detector. This
  **starves the camera/detector background threads of CPU/GIL time**, so YOLO can't run often
  enough and detection drops out. The original game (`game/game_loop.py`) and
  `flashlight_sandbox.py` were **deliberately optimized to avoid exactly this** (explicit code
  comments), which is why the sandbox detects **flawlessly** with the *same* camera + model.
- **Proof:** `python flashlight_sandbox.py --device 0` detects perfectly; `play_laser_arcade.py`
  with the **identical** CPU detector config detects terribly. Same camera, same model, same
  threads → the only difference is the **game process's render load**. Not the camera, not the
  model, not the color gate.
- **What went wrong in the debugging:** most of 06-17 was spent tuning the *output* of detection
  (green color gate, YOLO confidence, a camera "freeze" watchdog) — the **wrong layer**. Those
  changes were mostly harmless but could never fix a thread-starvation problem. The decisive
  sandbox-vs-game comparison came too late.
- **The fix when we resume:** optimize the arcade engine's render path the way
  `game_loop.py`/the sandbox already are (kill the per-frame `smoothscale`, cap render FPS to 30,
  throttle the detector via `MAX_DETECT_FPS`, cache per-frame surfaces) — edited in the
  **`laser-arcade` source** then re-vendored. Optionally move the detector to a separate
  **process** to defeat the GIL entirely.

## 1. System recap (so this stands alone)

- **Pipeline:** phone camera (IriunCam / openwebcam) → `game/capture.py` CaptureThread →
  `game/detector.py` DetectorThread (YOLO/OpenVINO; writes laser position to
  `shared.laser_state`) → the game reads `shared.laser_state`. Capture + detection run on
  **background threads**; the game render loop is the **main thread**. They share CPU under
  Python's **GIL**.
- **Games:** **thief** (dark maze, spotlight reveal, A* thief, escape doors) and **fruit ninja**
  (slice). They live in `arcade_game/` — a verbatim **vendored copy** of the separate
  **`laser-arcade`** repo (the source of truth). NEVER hand-edit `arcade_game/` for game/engine
  logic; edit `laser-arcade`, then re-vendor. Pipeline files
  (`capture/detector/calibration/config/main`) are edited directly in this repo.
- **Launchers:** `python play_laser_arcade.py` (new games, real laser), `python -m game.main`
  (original RL game, backup), `flashlight_sandbox.py` (isolated detector+spotlight tuner — the
  known-good detection baseline).
- **Machines:** **uni i7** = Intel Iris Xe iGPU, no NVIDIA — *the demo machine that must work*.
  **home PC** = RTX 3090 / CUDA — dev only. Profiles (`game/main.py`): `home`=CUDA/imgsz640,
  `uni`=OpenVINO-iGPU/imgsz320, `uni-cpu`=CPU `best.pt`/imgsz512 (added 06-17).
- **Confirmed timeline of the demos:** morning 06-17 on the i7 = worked great (before that day's
  changes); evening 06-17 on the **same i7** = unplayable (after the day's changes); 06-18
  testing on the home PC directly (no Remote Desktop). So the regression is real and bracketed by
  06-17's commits.

## 2. Commit timeline

| commit | when (06-) | what |
|---|---|---|
| 7549e39 | 16 | **Re-vendor games: animated thief + new Fruit Ninja** — first brought the heavy `arcade_game/` engine in |
| c2079ba / b972ae9 | 16 | play_laser_arcade entry point; vendor upgraded thief |
| f9ee46f | 17 00:23 | Laser launcher: Fruit Ninja + laser menu + live aim-nudge |
| 4af3c87 | 17 09:09 | Re-vendor Fruit Ninja blink fix |
| e9ec61b | 17 10:34 | Green color gate added (reject non-green blobs) |
| 79fb08b | 17 13:04 | **Tighten gate (H_LO 35→45)** + fruit ninja `show_pointer=False` |
| 0e426b7 | 17 15:14 | Thief: treasure hidden in the dark (game-source fix, pulled in) |
| 465d964 | 17 16:00 | Thief: bolt-when-lit + tougher catch (gameplay) |
| ea7ec94 | 17 16:13 | Loosen the gate again (undo the tighten) |
| e1acbd8 | 17 16:21 | Lower uni YOLO conf 0.20→0.10 |
| bf30080 | 17 17:09 | Camera "frozen feed" watchdog |
| 7508524 | 17 17:16 | **Add `uni-cpu` profile** (CPU best.pt) |
| 10a0d59 | 17 17:23 | Add `--imgsz` / `--render-fps` flags |

## 3. Symptoms, in the order they appeared

1. Spotlight jumps to the **gold treasure** (false detection).
2. After the green gate: **rare** treasure switching; fruit ninja "**starts good, degrades** over a run."
3. After **tightening** the gate: laser **blinks on/off**, thief uncatchable.
4. After loosening gate + lowering conf: **still blinking**, plus rare jumps to a random spot;
   **can't even select a menu tile**.
5. `--debug` log (decisive): `infer=30fps` steady, `gated=0` always, `detect` works ~8s then
   **flatlines at 0**, preview goes **grey**.
6. `flashlight_sandbox.py --device 0` detects **flawlessly** with the same camera → it's the
   **game process**, not the camera/model/gate.
7. `uni-cpu` "works but very low fps"; fruit ninja unsliceable → unplayable.

## 4. Every change — intent and honest verdict

| commit | intent | verdict |
|---|---|---|
| e9ec61b green gate | stop the treasure being detected | **band-aid** — real cure was hiding the treasure (0e426b7) |
| 79fb08b tighten gate H_LO→45 | kill rare treasure switch | **MISTAKE** — over-clipped the real (blown-out) laser → flicker |
| 79fb08b fruit ninja show_pointer off | stop white ring washing the green dot | reasonable, minor |
| 0e426b7 treasure hidden in dark | remove always-lit gold blob | **GOOD** (game-side, addresses a real cause) |
| 465d964 thief bolt + tougher catch | gameplay feel | unrelated to detection, but made catching harder → worsened the *perception* during the broken sessions |
| ea7ec94 loosen gate | undo the over-tighten | corrected the mistake; harmless |
| e1acbd8 conf 0.20→0.10 | latch a dim laser | harmless, **wrong layer** (`gated=0` proved gate/conf weren't the bottleneck) |
| bf30080 camera freeze watchdog | recover a frozen feed | **red herring** — the feed wasn't freezing (sandbox proved it); harmless safety net |
| 7508524 uni-cpu profile | dodge iGPU contention | **PARTIAL** — proved it's the game process; gave a slow-but-working path. The "iGPU contention" framing was incomplete: the real issue is CPU/GIL starvation, which hits the CPU path too |
| 10a0d59 imgsz / render-fps flags | speed up CPU detection | partial mitigation; the **render-fps cap is on the right track** (matches the real fix) |

## 5. Root cause (evidence)

A code investigation compared the arcade engine against the known-good `game/game_loop.py` and
`flashlight_sandbox.py`. Hypothesis **STRONGLY SUPPORTED**.

**The working code is deliberately optimized to NOT starve the detector thread:**
- `flashlight_sandbox.py`: *"`overlay` is a persistent SRCALPHA surface reused every frame (no
  per-frame allocation — that allocation was starving the YOLO thread of CPU)."* Pre-scales its
  grid **once**; caps `--render-fps` (default 30) and `--max-detect-fps` (default 12).
- `game/game_loop.py`: *"Full-screen dark overlay — allocated ONCE and reused every frame. A
  fresh 1920x1080 SRCALPHA surface per frame is a big needless alloc that competes with YOLO for
  CPU (the contention that stalled rendering)."* Caches the spotlight surface too.
- `game/detector.py`: *"Throttle inference rate so it doesn't run back-to-back and starve the
  render thread of CPU. Sleeping here releases CPU/GIL."* — honored only if `MAX_DETECT_FPS > 0`.

**The arcade engine reintroduces every one of those costs:**
- **#1 — full-screen `smoothscale` EVERY frame:** `arcade_game/arcade/display.py` `present()`
  does `pygame.transform.smoothscale(self.canvas, (draw_w, draw_h))` every frame whenever the
  window ≠ the 1920×1080 canvas (letterbox path). Bicubic scale of a 1080p surface, 60×/sec, no
  caching. **Biggest single cost.** (The sandbox does this **once** at init.)
- **#2 — `TARGET_FPS = 60`** (`arcade_game/arcade/config.py`) vs 30 in the optimized paths.
- **#3 — detector unthrottled:** the game profiles leave `MAX_DETECT_FPS = 0` (unlimited), so the
  detector runs back-to-back and never sleeps to release the GIL.
- **#4 — per-frame allocations:** thief = new 1920×1080 surface per shake frame and per
  capture-flash frame, edge-glow strips rebuilt per frame, a **new font object + 36 outline
  renders per frame** for the "ESCAPE!" banner; fruit ninja = a **new surface per juice particle
  per frame** and `pygame.transform.rotate` per fruit/bomb per frame.

**Mechanism / why ~0.5s dropouts:** the main thread holds the GIL doing this Python-level work
long enough that the capture + detector threads can't get CPU/GIL time; YOLO inference and frame
reads stall, so `shared.laser_state` goes stale → spotlight fades/blinks. It is **not** camera,
model, color, or confidence — those were the layers wrongly tuned. It hits **both** the iGPU and
CPU profiles because the bottleneck is main-thread CPU/GIL, not the inference device.

## 6. Why the fixes missed it (lessons)

- Treated the detector's **output** (color gate, confidence, camera) instead of the detector's
  **starvation**.
- Changed **many variables at once**, so cause/effect couldn't be attributed.
- The **decisive experiment** — run the sandbox and the game with the *same* detector config and
  compare — should have been **step 1**, not late. It instantly localizes the fault to the game
  process.
- Built a **camera freeze watchdog** for a freeze that wasn't happening (the sandbox proved the
  feed was fine on the same camera).
- **Home guidance mistake:** ran `--profile uni-cpu` (CPU) on the **home PC**, which has a CUDA
  3090 — that forces slow CPU inference and ignores the fast GPU. At home, use `--profile home`.

## 7. Fix plan for when we resume (prioritized)

**A. Engine optimization — the real fix.** Edit in the **`laser-arcade` source**, then re-vendor
into `arcade_game/`:
   1. **Kill the per-frame `smoothscale`** in `arcade/display.py`: make the pygame window exactly
      the canvas size so `present()` takes the direct-blit `scale==1` path (no scaling), OR use
      cheap `pygame.transform.scale` (nearest) instead of `smoothscale`, OR cache. **Do this
      first — likely the single biggest win.**
   2. **Cap render FPS to 30** (lower engine `TARGET_FPS`, or keep the launcher `--render-fps 30`
      default added in 10a0d59).
   3. **Throttle the detector:** set `MAX_DETECT_FPS ≈ 12–15` in the `uni`/`uni-cpu` profiles
      (`game/main.py`). The GIL-releasing sleep already exists in `game/detector.py`; the profile
      just needs to set the value.
   4. **Cache per-frame surfaces:** thief edge-glow strips, the "ESCAPE!" font (pre-render once),
      shake/flash full-screen surfaces; fruit ninja particle dots and rotated-sprite frames.
**B. Verify with `--debug`:** `detect=` fps must stay **steady (≥~12)** during play and never
collapse to 0.
**C. Home dev:** use `--profile home` (CUDA). Do **not** use `uni-cpu` at home.
**D. Long-term robustness:** run the detector in a **separate process** (multiprocessing, shared
frame via shared memory/queue) so rendering can never hold the GIL against it. The architectural
cure — makes detection immune to render load.
**E. Re-evaluate the band-aids:** the green gate + conf 0.10 are harmless but aimed at the wrong
problem. Once detection is steady, decide whether to keep the gate at a sensible middle setting
(it does reject the white HUD / red glow) or simplify. The over-tighten→loosen churn can be
collapsed.

## 8. Important context / gotchas

- **Vendoring:** engine + game edits go in `H:\Projects\laser-arcade`, then re-vendor (robocopy
  `/MIR` per subdir, exclude `__pycache__`) into `arcade_game/`.
- **Divergence (critical):** the thief **door/standstill** fixes, the **treasure-hide**, and the
  **bolt** change were **hand-applied** to the main vendored `arcade_game/games/thief/` and are
  **not all present in the `laser-arcade` source**. A blind thief re-vendor would regress them.
  **Reconcile the source first** before re-vendoring thief.
- `MAX_DETECT_FPS` lives in `game/config.py` (default 0 = unlimited); per-machine overrides go in
  the `PROFILES` dict in `game/main.py`.
- The per-frame `smoothscale` only fires when the fullscreen window ≠ exactly 1920×1080.
  **Check the i7's real fullscreen resolution** — if it isn't 1080p (very likely), `smoothscale`
  runs every frame.
- **Commit identities:** this repo = `TonmoyBishwas@users.noreply.github.com`; `laser-arcade` =
  `ttonmoy46@gmail.com`; both add the Claude co-author trailer. **Public repo — no secrets** (the
  Roboflow-key training notebook is untracked; never `git add -A`). **Green-free art rule**
  (bright green = laser only). Judge speed on the **projector**, not Remote Desktop.

## 9. "Is it fixed?" checklist for resume

- [ ] `flashlight_sandbox.py --device 0` still detects flawlessly (baseline sanity).
- [ ] After engine optimization, `play_laser_arcade.py --debug` shows a **steady `detect=` ≥ ~12
      fps** that does NOT collapse to 0 during play.
- [ ] Thief: spotlight tracks smoothly; a catch can be completed. Fruit ninja: fruits are sliceable.
- [ ] Validated on the **uni i7** specifically (the demo machine), not just the home PC.

## 10. Reflection

The debugging chased symptoms instead of isolating the layer. The pipeline has three layers —
camera feed, detector config (model/device/imgsz/gate/conf), and the game process that shares CPU
with the detector threads. The failure was in the **third** layer, but the day was spent editing
the **first two**, shipping many small speculative changes that muddied the signal and, in one
case (tightening the gate), briefly made things worse. The one cheap experiment that would have
pointed straight at the cause — "the sandbox detects fine with the same config, so what does the
game do differently?" — should have been run before touching anything. The good news: the cause
is now understood and concrete, the model and camera are fine, and the fix is a known, bounded
optimization pass on the engine.
