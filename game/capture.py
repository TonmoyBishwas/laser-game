"""
CaptureThread — reads frames from DroidCam USB virtual webcam.

Writes the latest frame into SharedState; always overwrites so the
detector thread never processes a stale frame.
"""

import threading
import time

import cv2

from game.config import DROIDCAM_DEVICE_INDEX


class CaptureThread(threading.Thread):
    def __init__(self, shared, stop_event: threading.Event, device_index: int = DROIDCAM_DEVICE_INDEX):
        super().__init__(name="CaptureThread", daemon=True)
        self._shared = shared
        self._stop_event = stop_event
        self._device = device_index

    def run(self) -> None:
        cap = self._open()
        print(f"[capture] opened device {self._device}")

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
        # CAP_ANY is the only backend that opens DroidCam on this PC.
        # MSMF opens a different (black) virtual device; DSHOW can't capture by index.
        cap = cv2.VideoCapture(self._device, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        raise RuntimeError(
            f"Cannot open camera device {self._device}. "
            "Make sure DroidCam USB is connected and the PC client is running."
        )
