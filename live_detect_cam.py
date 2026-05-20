r"""
Live laser detection from a webcam (IriunCam / any indexed camera).

Point the camera at a wall or desk (NOT the laptop screen, so there's no
infinity mirror), aim the laser, and watch whether YOLO boxes it. Use
this to sanity-check detection before mounting the camera on the tripod
facing the projector and running the actual game.

Defaults to the uni setup: camera index 2, CPU, imgsz=512, conf=0.20.

Usage
-----
  venv\Scripts\python.exe live_detect_cam.py                 (index 2, uni settings)
  venv\Scripts\python.exe live_detect_cam.py --device 0      (different camera)
  venv\Scripts\python.exe live_detect_cam.py --conf 0.10     (more sensitive)
  venv\Scripts\python.exe live_detect_cam.py --imgsz 640     (max recall)
  venv\Scripts\python.exe live_detect_cam.py --yolo cuda:0   (home GPU)

Keys in the window:
  q or ESC   quit
"""

import os
# Harmless on a normal webcam; helps some virtual cameras. Before cv2.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from game import config as cfg


BACKENDS = {"AUTO": cv2.CAP_ANY, "DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=2,
                   help="camera index (IriunCam on the uni laptop = 2)")
    p.add_argument("--backend", choices=list(BACKENDS), default="AUTO",
                   help="OpenCV camera backend")
    p.add_argument("--yolo", type=str, default="cpu",
                   help="YOLO device: cpu (uni) or cuda:0 (home)")
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=512,
                   help="YOLO input size (uni default 512)")
    p.add_argument("--display-width", type=int, default=1280,
                   help="max preview width; big feeds are downscaled to fit")
    args = p.parse_args()

    print(f"[detect] loading YOLO from {cfg.MODEL_PATH}")
    model = YOLO(cfg.MODEL_PATH)
    model.to(args.yolo)
    print(f"[detect] yolo={args.yolo} imgsz={args.imgsz} conf={args.conf}")

    print(f"[detect] opening camera index={args.device} backend={args.backend}")
    cap = cv2.VideoCapture(args.device, BACKENDS[args.backend])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[detect] FAILED to open camera index={args.device}. "
              f"Run find_cam.py to confirm the index, or try --backend MSMF.")
        return
    for _ in range(5):
        cap.read()

    win = "Laser detection  -  q/ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frames = 0
    dets = 0
    hit_frames = 0
    sum_conf = 0.0
    fail_reads = 0
    infer_ms_hist = []
    last_det_time = -10.0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_reads += 1
            if fail_reads > 200:
                print("[detect] too many failed reads — camera dropped.")
                break
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                break
            continue
        fail_reads = 0

        t_inf = time.time()
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device=args.yolo, verbose=False)[0]
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
            f"cam={args.device} yolo={args.yolo} imgsz={args.imgsz} conf={args.conf}",
            f"infer={avg_ms:.0f}ms ({fps_eff:.1f} fps)",
            f"dets={dets}  hit_pct={hit_pct:.1f}%  mean_conf={mean_conf:.2f}",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 28
            cv2.rectangle(annotated, (8, y - 20), (8 + 12 * len(line), y + 8),
                          (0, 0, 0), -1)
            cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (40, 255, 255), 2)

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
    print(f"[detect] {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} fps)")
    if frames:
        print(f"[detect] hit_pct={100.0 * hit_frames / frames:.1f}%  "
              f"mean_conf={(sum_conf / dets) if dets else 0.0:.2f}  "
              f"total_dets={dets}")
        print("[detect] If hit_pct is high when you aim at the laser, "
              "detection works — mount the tripod and run the game.")


if __name__ == "__main__":
    main()
