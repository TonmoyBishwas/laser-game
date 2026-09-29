"""
Laser Guard — NEW game layer (A* thief, random mazes, escape doors, custom art).

This runs the upgraded game that was built and play-tested in the separate
`laser-arcade` repo (with the mouse standing in for the laser) and is vendored
here under `arcade_game/`. It reuses the EXACT same camera -> calibration ->
detector pipeline as the original game; the only thing swapped is the input
source: a `LaserPointer` feeds the real laser into the game through the same
`Pointer` interface the mouse used in development.

The original RL game is untouched and remains the backup:

    python -m game.main           --profile uni --fullscreen        # backup (RL thief, fixed maze)

This launcher (laser only — the mouse is never used to play here):

    python play_laser_arcade.py   --profile uni --fullscreen        # menu: pick a game with the laser
    python play_laser_arcade.py   --profile uni --fullscreen --game thief        # straight into the thief
    python play_laser_arcade.py   --profile uni --fullscreen --game fruit_ninja  # straight into fruit ninja

Both share the same calibration file and the same --profile / --device / --url
options, so calibrating once works for either.

Aim correction: if the spotlight lands a little off from where the laser
actually points, nudge it live with the ARROW KEYS while playing (press the
direction you want the spot to move; it saves automatically), or pass a known
offset with --offset-x / --offset-y. The correction is stored in the
calibration file and applies to BOTH games and the RL backup.
"""

import os
# Match game.main: disable MSMF HW transforms BEFORE cv2 is imported (DroidCam fix).
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import argparse
import threading
import time

import pygame

_REPO = os.path.dirname(os.path.abspath(__file__))
# The vendored arcade engine + games (copied from the laser-arcade repo) live
# here. Insert at the FRONT so `import arcade` finds this local package.
sys.path.insert(0, os.path.join(_REPO, "arcade_game"))

import cv2  # noqa: F401  (keep the same import context as game.main)

from game import calibration as cal_mod
from game import capture as cap_mod
from game import detector as det_mod
from game.main import (SharedState, _open_camera, PROFILES, LASER_MODELS,
                       apply_laser_color, _run_debug_preview)
import game.config as cfg

# Vendored engine + games (from arcade_game/ on sys.path)
from arcade.display import enable_dpi_awareness
from arcade.scene import GameApp, MenuScene, Scene
from arcade.input import Pointer, PointerSource
from arcade.config import GAME_W, GAME_H
from games.thief.game import create as create_thief
from games.fruit_ninja.game import create as create_fruit

GAME_FACTORIES = {
    "thief": create_thief,
    "fruit_ninja": create_fruit,
}


class LaserPointer(PointerSource):
    """Feed the real laser into the game through the mouse's `Pointer` contract.

    `shared.laser_state.pos` is already in 1920x1080 game space (mapped by the
    calibration homography), so it drops straight onto the canvas with no extra
    transform. Staleness (LASER_STALE_SECS) becomes `Pointer.active`, exactly the
    way the original game_loop treated `laser_fresh` — when the beam goes stale
    the spotlight fades, same as moving the dev mouse off the window.
    """

    def __init__(self, shared: SharedState, stale_secs: float):
        self._shared = shared
        self._stale = stale_secs
        self._prev: tuple[float, float] | None = None
        self._last = (GAME_W / 2.0, GAME_H / 2.0)

    def poll(self) -> Pointer:
        with self._shared.laser_lock:
            ls = self._shared.laser_state

        active = ls is not None and (time.time() - ls.timestamp) < self._stale
        if active:
            x, y = ls.pos
            self._last = (x, y)
        else:
            x, y = self._last  # hold last spot; active=False makes the game ignore it

        vx = vy = 0.0
        if self._prev is not None:
            vx = x - self._prev[0]
            vy = y - self._prev[1]
        self._prev = (x, y)

        # Clamp to the canvas — the laser can only ever be reported on the surface.
        x = max(0.0, min(float(GAME_W), x))
        y = max(0.0, min(float(GAME_H), y))
        return Pointer(x, y, bool(active), vx, vy)


class AimNudgeScene(Scene):
    """Wrap a game scene to let the operator zero the laser aim live.

    The arrow keys nudge the calibration offset by a few game pixels and save it
    immediately, so the spotlight can be aligned to where the laser actually
    points without restarting. Press the direction you want the spot to MOVE.
    Everything else (gameplay, the R restart key, etc.) passes through to the
    real game untouched. This lives in the laser launcher (not the vendored
    engine), so it survives re-vendoring the games.
    """

    NUDGE = 4  # game pixels per arrow press

    def __init__(self, app, inner: Scene, calibration):
        super().__init__(app)
        self._inner = inner
        self._cal = calibration
        self._font = pygame.font.SysFont("monospace", 28, bold=True)
        self._show_until = 0.0

    @property
    def show_pointer(self) -> bool:               # mirror the wrapped game
        return getattr(self._inner, "show_pointer", True)

    def handle_event(self, event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (
            pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
            ox, oy = self._cal.offset
            if event.key == pygame.K_LEFT:    ox -= self.NUDGE
            elif event.key == pygame.K_RIGHT: ox += self.NUDGE
            elif event.key == pygame.K_UP:    oy -= self.NUDGE
            elif event.key == pygame.K_DOWN:  oy += self.NUDGE
            self._cal.offset = (ox, oy)           # same object the detector holds
            self._cal.save()                      # persist so it sticks next run
            self._show_until = time.time() + 4.0
            return
        self._inner.handle_event(event)

    def update(self, dt: float, pointer: Pointer) -> None:
        self._inner.update(dt, pointer)

    def draw(self, surface) -> None:
        self._inner.draw(surface)
        if time.time() < self._show_until:
            ox, oy = self._cal.offset
            txt = self._font.render(
                f"aim offset  x={ox:+.0f}  y={oy:+.0f}   (arrow keys nudge - saved)",
                True, (255, 230, 120))
            surface.blit(txt, (30, GAME_H - 50))


def main() -> None:
    enable_dpi_awareness()

    parser = argparse.ArgumentParser(
        description="Laser Guard — NEW game (A* thief + random mazes + escape doors)")
    parser.add_argument("--recalibrate", action="store_true", help="Force new 4-corner calibration")
    parser.add_argument("--device", type=int, default=None, help="Camera device index override")
    parser.add_argument("--url", type=str, default=None,
                        help="Camera stream URL (e.g. http://localhost:8090/video). "
                             "Overrides --device / profile camera index.")
    parser.add_argument("--fullscreen", action="store_true", help="Fullscreen on projector")
    parser.add_argument("--debug", action="store_true", help="YOLO preview window + detector stats")
    parser.add_argument("--profile", type=str, default="home", choices=list(PROFILES),
                        help="'home' = CUDA + imgsz 640 + cam 0; 'uni' = OpenVINO iGPU + imgsz 320")
    parser.add_argument("--game", type=str, default=None, choices=list(GAME_FACTORIES),
                        help="launch a game directly; omit to show the laser-pickable menu")
    parser.add_argument("--offset-x", type=float, default=None, dest="offset_x",
                        help="laser aim correction in game px (+x = move the spot RIGHT); persisted")
    parser.add_argument("--offset-y", type=float, default=None, dest="offset_y",
                        help="laser aim correction in game px (+y = move the spot DOWN); persisted")
    parser.add_argument("--no-color-gate", action="store_true", dest="no_color_gate",
                        help="disable the green-only filter (use only if a washed-out laser stops "
                             "being detected; the filter is what stops the treasure being detected)")
    parser.add_argument("--laser", type=str, default="green", choices=list(LASER_MODELS),
                        help="which laser pointer is in use: 'green' (default, YOLO26-nano) or "
                             "'red' (older YOLO26-S). Swaps the model and the colour gate.")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="override YOLO input size (overrides the profile). Lower = much faster "
                             "CPU inference: try 320 with --profile uni-cpu if detection is laggy "
                             "(the model was trained at 320). 256 is faster still; 512 is sharpest.")
    parser.add_argument("--render-fps", type=int, default=30, dest="render_fps",
                        help="cap the game render FPS (default 30). Lower frees the CPU for the "
                             "detector thread, so the laser tracks better on the CPU profile.")
    args = parser.parse_args()

    # Apply the device profile to game.config BEFORE the detector thread reads it.
    overrides = PROFILES[args.profile]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    print(f"[arcade] profile='{args.profile}' -> {overrides}")
    apply_laser_color(cfg, args.laser)
    if args.no_color_gate:
        cfg.LASER_COLOR_GATE = False
    if args.imgsz is not None:
        cfg.YOLO_IMGSZ = args.imgsz
        print(f"[arcade] YOLO_IMGSZ override -> {args.imgsz}")
    # Cap the engine render FPS so the render loop doesn't hog the CPU the
    # detector thread needs (set on the scene module, where GameApp.run reads it).
    import arcade.scene as _scene_mod
    _scene_mod.TARGET_FPS = max(10, args.render_fps)
    print(f"[arcade] render FPS cap -> {_scene_mod.TARGET_FPS}")
    if args.device is not None:
        cfg.DROIDCAM_DEVICE_INDEX = args.device

    camera_source = args.url if args.url else cfg.DROIDCAM_DEVICE_INDEX

    # ── Camera + calibration (identical to game.main) ──────────────────────────
    print("[arcade] opening camera for calibration…")
    cap = _open_camera(camera_source)
    try:
        calibration = cal_mod.Calibration.load_or_run(cap, force=args.recalibrate)
    except RuntimeError as e:
        cap.release()
        print(f"[arcade] calibration failed: {e}")
        return
    cap.release()
    time.sleep(0.3)

    # Apply a known aim correction from the CLI (absolute, in game px) and save it.
    # Otherwise keep whatever the calibration file already had; live arrow-key
    # nudges during play update and re-save it from there.
    if args.offset_x is not None or args.offset_y is not None:
        ox, oy = calibration.offset
        if args.offset_x is not None:
            ox = args.offset_x
        if args.offset_y is not None:
            oy = args.offset_y
        calibration.offset = (ox, oy)
        calibration.save()
    print(f"[arcade] laser aim offset = {calibration.offset} game px "
          f"(+x right, +y down) — nudge live with the arrow keys")

    # ── Threads ────────────────────────────────────────────────────────────────
    shared = SharedState()
    stop_event = threading.Event()
    capture_thread = cap_mod.CaptureThread(shared, stop_event, source=camera_source)
    detector_thread = det_mod.DetectorThread(shared, stop_event, calibration, debug=args.debug)
    capture_thread.start()
    detector_thread.start()

    preview_thread = None
    if args.debug:
        preview_thread = threading.Thread(
            target=_run_debug_preview, args=(shared, stop_event),
            name="PreviewThread", daemon=True)
        preview_thread.start()

    print("[arcade] threads started — launching game window")

    # ── Game (vendored arcade engine), driven by the real laser ────────────────
    # Wrap each game factory so the operator can zero the laser aim live with the
    # arrow keys; the menu launches these same wrapped factories.
    def _wrap(factory):
        return lambda app: AimNudgeScene(app, factory(app), calibration)

    registry = {name: _wrap(f) for name, f in GAME_FACTORIES.items()}

    app = GameApp(registry=registry, windowed=not args.fullscreen)
    app.pointer_source = LaserPointer(shared, cfg.LASER_STALE_SECS)

    # Hide the OS mouse cursor. On the projector it would otherwise be projected
    # and filmed by the camera, where the bright arrow can trigger false laser
    # detections that yank the spotlight to screen-centre. The laser is the only
    # input here, so the cursor serves no purpose.
    pygame.mouse.set_visible(False)

    # Launch a game directly, or show the laser-pickable menu (dwell to select).
    start = registry[args.game](app) if args.game else MenuScene(app)

    try:
        app.run(start)
    finally:
        print("[arcade] shutting down threads…")
        stop_event.set()
        capture_thread.join(timeout=3)
        detector_thread.join(timeout=3)
        if preview_thread is not None:
            preview_thread.join(timeout=3)
        print("[arcade] done")


if __name__ == "__main__":
    main()
