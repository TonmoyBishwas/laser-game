"""
DetectorThread — runs YOLO inference on the latest camera frame and
writes normalized game-space laser coordinates into SharedState.

Also publishes a debug stats snapshot and (when enabled) an annotated
frame for the optional preview window.
"""

import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from game import config as cfg
from game.calibration import Calibration


def _resolve_device(preferred: str) -> str:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        print(f"[detector] WARNING: requested {preferred} but CUDA is unavailable; falling back to CPU")
        return "cpu"
    if preferred.startswith("intel:gpu"):
        # OpenVINO iGPU path — fall back to CPU if no GPU plugin/device is visible.
        try:
            from openvino import Core
            if "GPU" not in Core().available_devices:
                print("[detector] Intel GPU not visible to OpenVINO; falling back to intel:cpu")
                return "intel:cpu"
        except Exception as e:
            print(f"[detector] OpenVINO GPU check failed ({e}); falling back to intel:cpu")
            return "intel:cpu"
    return preferred


class LaserState:
    __slots__ = ("pos", "conf", "timestamp", "camera_pos")

    def __init__(self, pos, conf, timestamp, camera_pos):
        self.pos = pos                # game-space (x, y) after homography
        self.conf = conf
        self.timestamp = timestamp
        self.camera_pos = camera_pos  # raw (cx, cy) in camera pixels


class DetectorStats:
    __slots__ = ("infer_fps", "detect_fps", "total_frames", "total_detections", "last_box")

    def __init__(self):
        self.infer_fps   = 0.0
        self.detect_fps  = 0.0
        self.total_frames     = 0
        self.total_detections = 0
        self.last_box = None   # (x1, y1, x2, y2) in camera pixels — for the preview window


class DetectorThread(threading.Thread):
    def __init__(self, shared, stop_event: threading.Event, calibration: Calibration, debug: bool = False):
        super().__init__(name="DetectorThread", daemon=True)
        self._shared = shared
        self._stop_event = stop_event
        self._cal = calibration
        self._debug = debug

    def run(self) -> None:
        device = _resolve_device(cfg.YOLO_DEVICE)
        model = YOLO(cfg.MODEL_PATH)
        # .to() only works on .pt PyTorch models. Exported formats (onnx,
        # openvino) take the device via predict() instead.
        if str(cfg.MODEL_PATH).lower().endswith(".pt"):
            model.to(device)
        print(f"[detector] model loaded ({cfg.MODEL_PATH}) on {device}, "
              f"conf={cfg.YOLO_CONF}, imgsz={cfg.YOLO_IMGSZ}")

        last_report = time.time()
        infer_count = 0
        detect_count = 0

        max_fps = getattr(cfg, "MAX_DETECT_FPS", 0) or 0
        min_interval = (1.0 / max_fps) if max_fps > 0 else 0.0

        while not self._stop_event.is_set():
            triggered = self._shared.frame_event.wait(timeout=0.05)
            if not triggered:
                continue

            with self._shared.frame_lock:
                frame = self._shared.latest_frame
                self._shared.frame_event.clear()

            if frame is None:
                continue

            iter_start = time.time()
            results = model.predict(
                frame,
                conf=cfg.YOLO_CONF,
                imgsz=cfg.YOLO_IMGSZ,
                device=device,
                verbose=False,
            )
            infer_count += 1

            boxes = results[0].boxes
            box_tuple = None
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best = int(np.argmax(confs))
                xyxy = boxes.xyxy[best].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                gx, gy = self._cal.transform(cx, cy)

                box_tuple = (x1, y1, x2, y2)
                detect_count += 1

                with self._shared.laser_lock:
                    self._shared.laser_state = LaserState(
                        pos=(gx, gy),
                        conf=float(confs[best]),
                        timestamp=time.time(),
                        camera_pos=(cx, cy),
                    )

            # Publish debug preview frame if enabled
            if self._debug:
                self._publish_preview(frame, box_tuple)

            # ── stats once per second ─────────────────────────────────────────
            now = time.time()
            if now - last_report >= 1.0:
                elapsed = now - last_report
                stats = self._shared.detector_stats
                stats.infer_fps  = infer_count / elapsed
                stats.detect_fps = detect_count / elapsed
                stats.total_frames     += infer_count
                stats.total_detections += detect_count
                stats.last_box = box_tuple
                print(
                    f"[detector] infer={stats.infer_fps:5.1f} fps  "
                    f"detect={stats.detect_fps:5.1f} fps  "
                    f"total_frames={stats.total_frames}  "
                    f"total_detections={stats.total_detections}"
                )
                infer_count = 0
                detect_count = 0
                last_report = now

            # Throttle inference rate so it doesn't run back-to-back and starve
            # the render thread of CPU. Sleeping here releases CPU/GIL.
            if min_interval > 0:
                spare = min_interval - (time.time() - iter_start)
                if spare > 0:
                    time.sleep(spare)

        print("[detector] stopped")

    def _publish_preview(self, frame: np.ndarray, box) -> None:
        annotated = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, ((x1 + x2) // 2, (y1 + y2) // 2), 6, (0, 255, 255), -1)
            cv2.putText(annotated, "LASER", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "no detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with self._shared.preview_lock:
            self._shared.preview_frame = annotated
