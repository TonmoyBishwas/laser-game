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


def _is_green_laser(frame_bgr, x1, y1, x2, y2) -> bool:
    """True if the detection box actually contains the green laser dot.

    The model fires on any bright blob, so projected game art (the gold treasure,
    white HUD text, the red edge glow) can be detected as a laser and drag the
    spotlight there. The laser is green, so we require the box to hold a real
    patch of green pixels. Thresholds come from config so the band can be tuned
    and the whole gate switched off live (cfg.LASER_COLOR_GATE / --no-color-gate).
    """
    h, w = frame_bgr.shape[:2]
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(x2) + 1), min(h, int(y2) + 1)
    area = (xi2 - xi1) * (yi2 - yi1)
    if area <= 0:
        return False
    hsv = cv2.cvtColor(frame_bgr[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Hue bands for the laser in use (cfg.LASER_COLOR): green is one band, red
    # wraps around OpenCV's 0/179 seam so it is two.
    if getattr(cfg, "LASER_COLOR", "green") == "red":
        bands = cfg.RED_GATE_H_BANDS
    else:
        bands = ((cfg.GREEN_GATE_H_LO, cfg.GREEN_GATE_H_HI),)
    hue_ok = np.zeros(h_ch.shape, dtype=bool)
    for lo, hi in bands:
        hue_ok |= (h_ch >= lo) & (h_ch <= hi)
    green = hue_ok & (s_ch >= cfg.GREEN_GATE_S_MIN) & (v_ch >= cfg.GREEN_GATE_V_MIN)
    count = int(np.count_nonzero(green))
    return count >= cfg.GREEN_GATE_MIN_PIXELS and (count / area) >= cfg.GREEN_GATE_MIN_FRACTION


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

        gate_on = getattr(cfg, "LASER_COLOR_GATE", True)
        color = getattr(cfg, "LASER_COLOR", "green")
        print(f"[detector] laser colour = {color}; colour gate {'ON' if gate_on else 'OFF'}"
              f"{f' (rejects non-{color} blobs like the treasure)' if gate_on else ' (--no-color-gate)'}")

        last_report = time.time()
        infer_count = 0
        detect_count = 0
        gated_count = 0   # boxes rejected this interval for not being green

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
            chosen = None   # (x1, y1, x2, y2, conf) of the accepted laser box
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                xyxy_all = boxes.xyxy.cpu().numpy()
                # Highest-confidence box first; skip any that fail the green test, so
                # bright game art can't masquerade as the laser even when it outscores
                # the real green dot. Without the gate, just take the top box.
                for i in np.argsort(confs)[::-1]:
                    bx = xyxy_all[i]
                    x1, y1, x2, y2 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
                    if gate_on and not _is_green_laser(frame, x1, y1, x2, y2):
                        gated_count += 1
                        continue
                    chosen = (x1, y1, x2, y2, float(confs[i]))
                    break

            if chosen is not None:
                x1, y1, x2, y2, conf = chosen
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                gx, gy = self._cal.transform(cx, cy)

                box_tuple = (x1, y1, x2, y2)
                detect_count += 1

                with self._shared.laser_lock:
                    self._shared.laser_state = LaserState(
                        pos=(gx, gy),
                        conf=conf,
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
                    f"gated={gated_count:3d}  "
                    f"total_frames={stats.total_frames}  "
                    f"total_detections={stats.total_detections}"
                )
                infer_count = 0
                detect_count = 0
                gated_count = 0
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
