"""
Entry point for the Laser Guard game.

Usage
-----
  python -m game.main                       # normal run
  python -m game.main --recalibrate         # force new calibration
  python -m game.main --device 1            # use camera device index 1
  python -m game.main --fullscreen          # fullscreen on projector
  python -m game.main --debug               # YOLO preview + HUD stats + Q-bars
  python -m game.main --policy other.npz    # load a different Q-table
  python -m game.main --profile uni         # CPU + imgsz=512 + cam index 2 (uni i7 / demo)
  python -m game.main --profile home        # CUDA + imgsz=640 + cam index 0 (default, dev PC)
"""

import os
# Fix DroidCam green/garbled feed: disable MSMF hardware transforms BEFORE
# cv2 (or any game submodule that imports it) is loaded.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Device profiles — applied to game.config BEFORE detector imports anything from it.
PROFILES = {
    "home": {"YOLO_DEVICE": "cuda:0", "YOLO_IMGSZ": 640, "YOLO_CONF": 0.25,
             "DROIDCAM_DEVICE_INDEX": 0},
    # uni laptop is an i7/16GB (no GPU) running IriunCam at index 2.
    # imgsz 512 (vs the old i3-era 320) gives much better small-laser recall
    # and the i7 has the headroom. Drop to 416/320 if detection feels laggy.
    "uni":  {"YOLO_DEVICE": "cpu",    "YOLO_IMGSZ": 512, "YOLO_CONF": 0.20,
             "DROIDCAM_DEVICE_INDEX": 2},
}

import argparse
import threading
import time

import cv2

from game import calibration as cal_mod
from game import capture as cap_mod
from game import detector as det_mod
from game import game_loop
from game.config import POLICY_PATH
from game.detector import DetectorStats
from game.maze import Maze
from game.maze_layouts import DEFAULT
from game.rl_policy import QPolicy


class SharedState:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock   = threading.Lock()
        self.frame_event  = threading.Event()

        self.laser_state = None
        self.laser_lock  = threading.Lock()

        self.detector_stats = DetectorStats()

        self.preview_frame = None
        self.preview_lock  = threading.Lock()


def _open_camera(source) -> cv2.VideoCapture:
    # source is an int device index OR an MJPEG-over-HTTP URL string
    # (openwebcam app). A URL is read natively by OpenCV's FFmpeg backend;
    # an int index goes through CAP_ANY.
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        backend = "URL"
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)
        backend = "CAP_ANY"
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[main] camera {source!r} opened via {backend}")
            return cap
    cap.release()
    raise RuntimeError(
        f"Cannot open camera source {source!r}. "
        "USB index: check the webcam is connected. "
        "Stream URL: confirm the phone app is streaming (test with "
        "test_mobile_cam.py first)."
    )


def _run_debug_preview(shared, stop_event: threading.Event) -> None:
    win = "DEBUG — camera + YOLO boxes  (q to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 600)
    while not stop_event.is_set():
        with shared.preview_lock:
            frame = shared.preview_frame
        if frame is not None:
            cv2.imshow(win, frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyWindow(win)


def _load_policy(path: str) -> QPolicy:
    if not os.path.exists(path):
        print(f"[main] WARNING: policy {path} not found. Run train_thief.py first.")
        print(f"[main] Using untrained policy (thief will move randomly).")
        return QPolicy(epsilon=0.0)
    policy = QPolicy.load(path)
    maze = Maze(DEFAULT)
    meta = policy.metadata
    sig = meta.get("maze_signature")
    if sig and sig != maze.signature:
        print(
            f"[main] WARNING: policy was trained on maze {sig} but current maze is {maze.signature}.\n"
            "       The policy may behave poorly. Retrain with train_thief.py."
        )
    print(
        f"[main] policy episodes={meta.get('episodes')}  "
        f"win_rate={meta.get('final_win_rate', 0):.2%}  "
        f"caught_rate={meta.get('final_caught_rate', 0):.2%}  "
        f"trained_at={meta.get('trained_at')}"
    )
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Laser Guard — find the thief in the dark")
    parser.add_argument("--recalibrate", action="store_true", help="Force new 4-corner calibration")
    parser.add_argument("--device",      type=int, default=None, help="Camera device index override")
    parser.add_argument("--url",         type=str, default=None,
                        help="Camera stream URL (e.g. http://localhost:8090/video from the "
                             "openwebcam app). Overrides --device / profile camera index.")
    parser.add_argument("--fullscreen",  action="store_true",    help="Fullscreen on projector")
    parser.add_argument("--debug",       action="store_true",    help="YOLO preview window + HUD stats + Q-bars")
    parser.add_argument("--policy",      type=str, default=POLICY_PATH, help="Path to Q-table NPZ")
    parser.add_argument("--profile",     type=str, default="home", choices=list(PROFILES),
                        help="'home' = CUDA + imgsz 640 + cam 0; 'uni' = CPU + imgsz 512 + cam 2")
    args = parser.parse_args()

    # Apply the device profile BEFORE the detector thread reads cfg.YOLO_*.
    import game.config as cfg
    overrides = PROFILES[args.profile]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    print(f"[main] profile='{args.profile}' -> {overrides}")

    if args.device is not None:
        cfg.DROIDCAM_DEVICE_INDEX = args.device

    from game.config import DROIDCAM_DEVICE_INDEX

    # A --url stream (openwebcam) takes priority over any device index.
    camera_source = args.url if args.url else DROIDCAM_DEVICE_INDEX

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = _load_policy(args.policy)

    # ── Camera + calibration ──────────────────────────────────────────────────
    print("[main] opening camera for calibration…")
    cap = _open_camera(camera_source)
    try:
        calibration = cal_mod.Calibration.load_or_run(cap, force=args.recalibrate)
    except RuntimeError as e:
        cap.release()
        print(f"[main] calibration failed: {e}")
        return
    cap.release()
    time.sleep(0.3)

    # ── Threads ───────────────────────────────────────────────────────────────
    shared     = SharedState()
    stop_event = threading.Event()

    capture_thread  = cap_mod.CaptureThread(shared, stop_event, source=camera_source)
    detector_thread = det_mod.DetectorThread(shared, stop_event, calibration, debug=args.debug)
    capture_thread.start()
    detector_thread.start()

    preview_thread = None
    if args.debug:
        preview_thread = threading.Thread(
            target=_run_debug_preview, args=(shared, stop_event),
            name="PreviewThread", daemon=True,
        )
        preview_thread.start()

    print("[main] threads started — launching game window")

    try:
        game_loop.run(shared, policy, fullscreen=args.fullscreen, debug=args.debug)
    finally:
        print("[main] shutting down threads…")
        stop_event.set()
        capture_thread.join(timeout=3)
        detector_thread.join(timeout=3)
        if preview_thread is not None:
            preview_thread.join(timeout=3)
        print("[main] done")


if __name__ == "__main__":
    main()
