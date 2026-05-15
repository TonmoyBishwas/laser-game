"""
Live YOLO detection tuner — adjust thresholds with sliders and see the result instantly.

Modes
-----
  (1) RECORD a clip first, then tune offline (RECOMMENDED if camera sees its own preview):

      Step 1 — record 30 seconds of yourself pointing the laser:
        .\venv\Scripts\python.exe tune_detect.py --record clip.mp4 --duration 30

      Step 2 — tune against the recorded clip (loops forever):
        .\venv\Scripts\python.exe tune_detect.py --source clip.mp4

  (2) LIVE tuning straight from the camera (only works when the camera does NOT
      see the tuner window itself — i.e. when you have a separate display):
        .\venv\Scripts\python.exe tune_detect.py
        .\venv\Scripts\python.exe tune_detect.py --device 0
        .\venv\Scripts\python.exe tune_detect.py --source 0       # same as above

Keys in the preview window:
  q   quit
  p   pause / resume
  s   save current sliders to detector_tuning.json
  r   reset sliders to defaults
  +/- nudge confidence by 0.01
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = r"runs\detect\laser_detector_run\weights\best.pt"
TUNING_FILE = Path("detector_tuning.json")

IMGSZ_OPTIONS  = [320, 416, 512, 640, 800]
PREPROC_NAMES  = ["off", "brighter", "darker", "contrast+", "CLAHE"]

WINDOW = "YOLO laser detection tuner"


# ── Source helpers ─────────────────────────────────────────────────────────────

def open_source(source: str | int) -> tuple[cv2.VideoCapture, bool]:
    """Returns (cap, is_video_file). For video files, frames loop forever."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {idx}")
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Camera {idx} opened but cannot read frames")
        print(f"[tune] source = LIVE camera index {idx}")
        return cap, False

    # file path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {path}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"[tune] source = file {path}  ({nframes} frames @ {fps:.1f} fps)")
    return cap, True


def read_frame(cap: cv2.VideoCapture, is_video: bool) -> np.ndarray | None:
    ok, frame = cap.read()
    if ok:
        return frame
    if is_video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        return frame if ok else None
    return None


# ── Record mode ────────────────────────────────────────────────────────────────

def record_clip(device: int, out_path: str, duration: float) -> None:
    """
    Capture `duration` seconds from the camera straight to `out_path` (mp4).
    Shows minimal terminal countdown — no preview window, so no infinity mirror.
    """
    cap = cv2.VideoCapture(device, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {device}")

    # Warm up + measure FPS
    for _ in range(10):
        cap.read()
    fps_meas = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps_meas, (w, h))

    print(f"[record] writing {out_path}  ({w}x{h} @ {fps_meas:.1f} fps  for {duration}s)")
    print("[record] move the laser around — try the center, edges, etc.")
    print("[record] 3...")
    time.sleep(1)
    print("[record] 2...")
    time.sleep(1)
    print("[record] 1...")
    time.sleep(1)
    print("[record] GO!")

    t0 = time.time()
    nframes = 0
    while time.time() - t0 < duration:
        ok, frame = cap.read()
        if ok and frame is not None:
            out.write(frame)
            nframes += 1
        else:
            time.sleep(0.01)
        # countdown
        remaining = duration - (time.time() - t0)
        print(f"\r[record] {remaining:4.1f}s left   frames={nframes}", end="", flush=True)
    print()

    out.release()
    cap.release()
    print(f"[record] done. wrote {nframes} frames to {out_path}")
    print(f"[record] tune against it with:  .\\venv\\Scripts\\python.exe tune_detect.py --source {out_path}")


# ── Pre-processing ─────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return frame
    if mode == 1:
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=40)
    if mode == 2:
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=-30)
    if mode == 3:
        return cv2.convertScaleAbs(frame, alpha=1.4, beta=0)
    if mode == 4:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return frame


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray, boxes_xyxy, confs, show_min_conf: float) -> int:
    shown = 0
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs):
        if c < show_min_conf:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        green = int(255 * min(1.0, c))
        red   = int(255 * (1.0 - min(1.0, c)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, green, red), 2)
        cv2.putText(frame, f"{c:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, green, red), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
        shown += 1
    return shown


def draw_hud(frame: np.ndarray, *, conf: float, iou: float, imgsz: int,
             show_min_conf: float, preproc: str,
             total_dets: int, shown_dets: int, fps: float,
             paused: bool, source_label: str) -> None:
    bar_h = 130
    bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
    bar[:] = (15, 15, 25)
    lines = [
        f"src={source_label}  conf={conf:.2f}  iou={iou:.2f}  imgsz={imgsz}  show>={show_min_conf:.2f}  preproc={preproc}",
        f"detections raw={total_dets}  shown={shown_dets}  fps={fps:5.1f}  {'[PAUSED]' if paused else ''}",
        "keys: q quit | p pause | s save | r reset | +/- conf fine",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(bar, ln, (12, 28 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 220, 255), 2 if i == 0 else 1)
    out = np.vstack([bar, frame])
    cv2.imshow(WINDOW, out)


def save_tuning(conf: float, iou: float, imgsz: int, preproc: int) -> None:
    payload = {
        "conf": conf,
        "iou":  iou,
        "imgsz": imgsz,
        "preproc": preproc,
        "preproc_name": PREPROC_NAMES[preproc],
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    TUNING_FILE.write_text(json.dumps(payload, indent=2))
    print(f"[tune] saved -> {TUNING_FILE.resolve()}")
    print(f"       {payload}")


# ── Main tune loop ─────────────────────────────────────────────────────────────

def run_tuner(source: str | int, weights: str, use_cuda: bool) -> None:
    device = "cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"[tune] device={device}")
    model = YOLO(weights)
    model.to(device)
    print(f"[tune] loaded {weights}")

    cap, is_video = open_source(source)
    source_label = f"file:{Path(str(source)).name}" if is_video else f"camera:{source}"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1100, 820)

    defaults = {"conf": 25, "iou": 45, "imgsz_idx": 3, "show_min": 25, "preproc": 0}
    if TUNING_FILE.exists():
        try:
            data = json.loads(TUNING_FILE.read_text())
            defaults["conf"]      = int(round(data.get("conf", 0.25) * 100))
            defaults["iou"]       = int(round(data.get("iou",  0.45) * 100))
            defaults["imgsz_idx"] = IMGSZ_OPTIONS.index(int(data.get("imgsz", 640)))
            defaults["preproc"]   = int(data.get("preproc", 0))
            print(f"[tune] restored sliders from {TUNING_FILE}")
        except Exception as e:
            print(f"[tune] could not parse {TUNING_FILE}: {e}")

    cv2.createTrackbar("conf x100",  WINDOW, defaults["conf"],     100, lambda v: None)
    cv2.createTrackbar("iou  x100",  WINDOW, defaults["iou"],      100, lambda v: None)
    cv2.createTrackbar("imgsz idx",  WINDOW, defaults["imgsz_idx"], 4,  lambda v: None)
    cv2.createTrackbar("show conf",  WINDOW, defaults["show_min"], 100, lambda v: None)
    cv2.createTrackbar("preproc",    WINDOW, defaults["preproc"],   4,  lambda v: None)

    paused = False
    last_frame = None
    last_t = time.time()
    fps_smooth = 0.0

    while True:
        conf  = cv2.getTrackbarPos("conf x100",  WINDOW) / 100.0
        iou   = cv2.getTrackbarPos("iou  x100",  WINDOW) / 100.0
        idx   = cv2.getTrackbarPos("imgsz idx",  WINDOW)
        imgsz = IMGSZ_OPTIONS[max(0, min(idx, len(IMGSZ_OPTIONS) - 1))]
        show_min = cv2.getTrackbarPos("show conf", WINDOW) / 100.0
        preproc  = cv2.getTrackbarPos("preproc",   WINDOW)

        if not paused:
            frame = read_frame(cap, is_video)
            if frame is None:
                continue
            last_frame = frame.copy()

        if last_frame is None:
            continue
        frame = last_frame.copy()

        # Inference
        infer_in = preprocess(frame, preproc)
        results = model.predict(
            infer_in, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            xyxy, confs = [], []
        else:
            xyxy  = boxes.xyxy.cpu().numpy().tolist()
            confs = boxes.conf.cpu().numpy().tolist()

        total = len(confs)
        shown = draw_boxes(frame, xyxy, confs, show_min_conf=show_min)

        now = time.time()
        inst_fps = 1.0 / max(now - last_t, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
        last_t = now

        draw_hud(frame, conf=conf, iou=iou, imgsz=imgsz, show_min_conf=show_min,
                 preproc=PREPROC_NAMES[preproc], total_dets=total, shown_dets=shown,
                 fps=fps_smooth, paused=paused, source_label=source_label)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"[tune] {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord("s"):
            save_tuning(conf, iou, imgsz, preproc)
        elif key == ord("r"):
            cv2.setTrackbarPos("conf x100", WINDOW, 25)
            cv2.setTrackbarPos("iou  x100", WINDOW, 45)
            cv2.setTrackbarPos("imgsz idx", WINDOW, 3)
            cv2.setTrackbarPos("show conf", WINDOW, 25)
            cv2.setTrackbarPos("preproc",   WINDOW, 0)
            print("[tune] reset sliders to defaults")
        elif key in (ord("+"), ord("=")):
            cur = cv2.getTrackbarPos("conf x100", WINDOW)
            cv2.setTrackbarPos("conf x100", WINDOW, min(100, cur + 1))
        elif key == ord("-"):
            cur = cv2.getTrackbarPos("conf x100", WINDOW)
            cv2.setTrackbarPos("conf x100", WINDOW, max(0, cur - 1))

    cap.release()
    cv2.destroyAllWindows()
    print("[tune] bye.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0",
                   help="camera index (0,1,...) OR path to .mp4 video file")
    p.add_argument("--device", type=int, default=None,
                   help="alias for --source <int> (kept for backward compat)")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--cuda", action="store_true", default=True)
    p.add_argument("--record", type=str, default=None,
                   help="If set, record from camera into this mp4 path and exit")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds to record (with --record)")
    args = p.parse_args()

    # Record mode
    if args.record:
        dev = args.device if args.device is not None else 0
        record_clip(device=dev, out_path=args.record, duration=args.duration)
        return

    # Tuning mode
    source = args.source
    if args.device is not None:           # backward compat
        source = str(args.device)
    run_tuner(source=source, weights=args.weights, use_cuda=args.cuda)


if __name__ == "__main__":
    main()
