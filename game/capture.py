"""
CaptureThread — reads frames from the camera source and publishes the
latest frame to SharedState.

The source is either:
  * an int device index (USB / virtual webcam), or
  * an MJPEG-over-HTTP URL string, e.g. "http://localhost:8090/video"
    served by the openwebcam phone app (see MOBILE_CAM_APP_NOTES.md).

Always overwrites the latest frame so the detector never processes a
stale frame.
"""

import threading
import time

import cv2
import numpy as np

from game.config import DROIDCAM_DEVICE_INDEX

# Frozen-feed watchdog. Phone webcam apps (IriunCam etc.) can drop the stream
# yet keep the virtual camera "open", feeding a flat grey/static frame at full
# fps — so reads still succeed but YOLO sees no laser and detection dies until
# the program is restarted. A LIVE camera always has sensor noise, so truly
# byte-stable frames mean the stream has stalled. If frames stop changing for
# this long, reopen the camera to try to re-establish the stream.
FREEZE_REOPEN_SECS = 2.5
_FREEZE_PIXEL_EPS  = 4     # per-pixel level change (0-255) that still counts as "no change"


def _frame_signature(frame):
    """Cheap, noise-tolerant fingerprint of a frame: a small downsampled grid.

    A live sensor jitters by a few levels every frame, so comparing this grid
    catches a genuinely frozen feed without false-firing on a static-but-live
    scene (e.g. the laser resting on a dark maze)."""
    return frame[::40, ::40].astype(np.int16)


class CaptureThread(threading.Thread):
    def __init__(self, shared, stop_event: threading.Event, source=DROIDCAM_DEVICE_INDEX):
        super().__init__(name="CaptureThread", daemon=True)
        self._shared = shared
        self._stop_event = stop_event
        self._source = source

    def run(self) -> None:
        cap = self._open()
        print(f"[capture] opened source {self._source!r}")

        prev_sig = None
        last_change = time.time()

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[capture] read failed — retrying in 0.5s")
                cap.release()
                time.sleep(0.5)
                cap = self._open()
                prev_sig, last_change = None, time.time()
                continue

            # Frozen-feed watchdog: a live camera is never byte-stable (sensor
            # noise), so if the frame stops changing the phone stream has stalled
            # — reopen to recover instead of feeding YOLO a dead grey image.
            sig = _frame_signature(frame)
            now = time.time()
            if prev_sig is not None and np.max(np.abs(sig - prev_sig)) <= _FREEZE_PIXEL_EPS:
                if now - last_change >= FREEZE_REOPEN_SECS:
                    print(f"[capture] feed frozen for {now - last_change:.1f}s "
                          f"(phone stream likely dropped) — reopening camera")
                    cap.release()
                    time.sleep(0.3)
                    cap = self._open()
                    prev_sig, last_change = None, time.time()
                    continue
            else:
                last_change = now
            prev_sig = sig

            with self._shared.frame_lock:
                self._shared.latest_frame = frame
                self._shared.frame_event.set()

        cap.release()
        print("[capture] stopped")

    def _open(self) -> cv2.VideoCapture:
        # A URL string is read natively by OpenCV's FFmpeg backend (MJPEG over
        # HTTP — the openwebcam app). An int index goes through CAP_ANY, the
        # only backend that opened the USB virtual webcams on these machines.
        if isinstance(self._source, str):
            cap = cv2.VideoCapture(self._source)
        else:
            cap = cv2.VideoCapture(self._source, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        raise RuntimeError(
            f"Cannot open camera source {self._source!r}. "
            "USB index: check the webcam is connected. "
            "Stream URL: confirm the phone app is streaming (test with "
            "test_mobile_cam.py first)."
        )
