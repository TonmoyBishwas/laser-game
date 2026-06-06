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

from game.config import DROIDCAM_DEVICE_INDEX


class CaptureThread(threading.Thread):
    def __init__(self, shared, stop_event: threading.Event, source=DROIDCAM_DEVICE_INDEX):
        super().__init__(name="CaptureThread", daemon=True)
        self._shared = shared
        self._stop_event = stop_event
        self._source = source

    def run(self) -> None:
        cap = self._open()
        print(f"[capture] opened source {self._source!r}")

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[capture] read failed — retrying in 0.5s")
                cap.release()
                time.sleep(0.5)
                cap = self._open()
                continue

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
