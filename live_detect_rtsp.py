r"""
Live laser detection over an RTSP stream.

Runs YOLO on the phone's RTSP feed with the same settings the demo
laptop will use (uni profile: CPU, imgsz=320, conf=0.20), draws
detection boxes + live stats, so you can confirm the laser is being
detected over the network stream before wiring it into the game.

The RTSP URL comes from rtsp_url.py (edit it once). A --url argument
overrides it.

Usage
-----
  1. Edit RTSP_URL in rtsp_url.py, then:
       venv\Scripts\python.exe live_detect_rtsp.py
  2. Or override / tune:
       venv\Scripts\python.exe live_detect_rtsp.py --url rtsp://...
       venv\Scripts\python.exe live_detect_rtsp.py --conf 0.10 --imgsz 640
       venv\Scripts\python.exe live_detect_rtsp.py --device cuda:0   # home GPU test

Keys in the window:
  q or ESC   quit
"""

import os
# Force RTSP over TCP for stability. MUST be set before cv2 is imported.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from game import config as cfg
from rtsp_url import RTSP_URL


def open_rtsp(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, default="",
                   help="RTSP URL override; defaults to RTSP_URL in rtsp_url.py")
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO confidence threshold (uni default 0.20)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="YOLO input size (uni default 320, home default 640)")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu, cuda:0, etc. (uni default cpu)")
    p.add_argument("--display-width", type=int, default=1280,
                   help="max preview width; big streams are downscaled to fit")
    args = p.parse_args()

    url = args.url.strip() or RTSP_URL.strip()
    if not url:
        print("[detect] no URL. Set RTSP_URL in rtsp_url.py or pass --url rtsp://...")
        return

    print(f"[detect] loading YOLO from {cfg.MODEL_PATH}")
    model = YOLO(cfg.MODEL_PATH)
    model.to(args.device)
    print(f"[detect] device={args.device} imgsz={args.imgsz} conf={args.conf}")

    print(f"[detect] opening: {url}")
    print("[detect] (RTSP can take 5-10s to connect — be patient)...")
    cap = open_rtsp(url)
    if not cap.isOpened():
        print("[detect] FAILED to open the stream. Run test_rtsp.py first to "
              "confirm the URL is live.")
        return

    win = "RTSP detection  -  q/ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frames = 0
    dets = 0
    sum_conf = 0.0
    hit_frames = 0
    fail_reads = 0
    infer_ms_hist = []
    last_det_time = -10.0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_reads += 1
            if fail_reads > 200:
                print("[detect] too many failed reads — stream dropped.")
                break
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                break
            continue
        fail_reads = 0

        t_inf = time.time()
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        infer_ms = (time.time() - t_inf) * 1000.0
        infer_ms_hist.append(infer_ms)
        if len(infer_ms_hist) > 60:
            infer_ms_hist.pop(0)

        n = len(result.boxes)
        frames += 1
        if n > 0:
            dets += n
            hit_frames += 1
            sum_conf += float(result.boxes.conf.sum().item())
            last_det_time = time.time()

        annotated = frame.copy()
        if n > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                c = float(box.conf[0].item())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{c:.2f}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        mean_conf = (sum_conf / dets) if dets > 0 else 0.0
        hit_pct = 100.0 * hit_frames / max(frames, 1)
        avg_ms = float(np.mean(infer_ms_hist)) if infer_ms_hist else 0.0
        fps_eff = 1000.0 / max(avg_ms, 1e-3)
        lines = [
            f"device={args.device} imgsz={args.imgsz} conf={args.conf}",
            f"infer={avg_ms:.0f}ms ({fps_eff:.1f} fps)",
            f"dets={dets}  hit_pct={hit_pct:.1f}%  mean_conf={mean_conf:.2f}",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 28
            cv2.rectangle(annotated, (8, y - 20), (8 + 12 * len(line), y + 8),
                          (0, 0, 0), -1)
            cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (40, 255, 255), 2)

        # detection indicator: green dot top-right that lights on a recent hit
        since = time.time() - last_det_time
        col = (0, 230, 0) if since < 0.3 else (60, 60, 60)
        cv2.circle(annotated, (annotated.shape[1] - 40, 40), 22, col, -1)
        cv2.circle(annotated, (annotated.shape[1] - 40, 40), 22, (255, 255, 255), 2)

        disp = annotated
        if disp.shape[1] > args.display_width:
            scale = args.display_width / disp.shape[1]
            disp = cv2.resize(annotated, (int(disp.shape[1] * scale),
                                          int(disp.shape[0] * scale)))
        cv2.imshow(win, disp)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = max(time.time() - t0, 1e-3)
    print()
    print(f"[detect] {frames} frames in {elapsed:.1f}s "
          f"({frames / elapsed:.1f} fps end-to-end)")
    if frames:
        print(f"[detect] hit_pct={100.0 * hit_frames / frames:.1f}%  "
              f"mean_conf={(sum_conf / dets) if dets else 0.0:.2f}  "
              f"total_dets={dets}")


if __name__ == "__main__":
    main()
