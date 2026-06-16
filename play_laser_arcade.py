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

This new game:

    python play_laser_arcade.py   --profile uni --fullscreen        # A* thief, random mazes, doors, art

Both share the same calibration file and the same --profile / --device / --url
options, so calibrating once works for either.
"""

import os
# Match game.main: disable MSMF HW transforms BEFORE cv2 is imported (DroidCam fix).
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import argparse
import threading
import time

_REPO = os.path.dirname(os.path.abspath(__file__))
# The vendored arcade engine + games (copied from the laser-arcade repo) live
# here. Insert at the FRONT so `import arcade` finds this local package.
sys.path.insert(0, os.path.join(_REPO, "arcade_game"))

import cv2  # noqa: F401  (keep the same import context as game.main)

from game import calibration as cal_mod
from game import capture as cap_mod
from game import detector as det_mod
from game.main import SharedState, _open_camera, PROFILES, _run_debug_preview
import game.config as cfg

# Vendored engine + game (from arcade_game/ on sys.path)
from arcade.display import enable_dpi_awareness
from arcade.scene import GameApp
from arcade.input import Pointer, PointerSource
from arcade.config import GAME_W, GAME_H
from games.thief.game import create as create_thief


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
    args = parser.parse_args()

    # Apply the device profile to game.config BEFORE the detector thread reads it.
    overrides = PROFILES[args.profile]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    print(f"[arcade] profile='{args.profile}' -> {overrides}")
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
    app = GameApp(registry={"thief": create_thief}, windowed=not args.fullscreen)
    app.pointer_source = LaserPointer(shared, cfg.LASER_STALE_SECS)

    # Hide the OS mouse cursor. On the projector it would otherwise be projected
    # and filmed by the camera, where the bright arrow can trigger false laser
    # detections that yank the spotlight to screen-centre. The laser is the only
    # input here, so the cursor serves no purpose.
    import pygame
    pygame.mouse.set_visible(False)

    try:
        app.run(create_thief(app))
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
