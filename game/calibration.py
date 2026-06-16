"""
One-time perspective calibration.

Usage
-----
  cal = Calibration.load_or_run(cap)
  game_xy = cal.transform(camera_x, camera_y)
"""

import json
import os
import cv2
import numpy as np

from game.config import GAME_W, GAME_H, CALIBRATION_FILE

_WINDOW = "Calibration"
_CORNER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
_GAME_CORNERS = np.array(
    [[0, 0], [GAME_W, 0], [GAME_W, GAME_H], [0, GAME_H]], dtype=np.float32
)


class Calibration:
    def __init__(self, H: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)):
        self._H = H
        # Aim correction applied AFTER the homography, in game-space pixels.
        # The homography maps the camera dot to game space; a small systematic
        # bias still remains (slightly off-centre calibration clicks, or the
        # bright laser bloom shifting the box centroid), which shows up as the
        # spotlight landing a little off from where the laser actually points.
        # This constant nudge cancels it. +x shifts the reported spot RIGHT,
        # +y shifts it DOWN. Tunable live (arrow keys) and persisted below.
        self.offset = (float(offset[0]), float(offset[1]))

    def transform(self, cx: float, cy: float) -> tuple[float, float]:
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)[0][0]
        return float(out[0]) + self.offset[0], float(out[1]) + self.offset[1]

    def save(self, path: str = CALIBRATION_FILE) -> None:
        with open(path, "w") as f:
            json.dump({"H": self._H.tolist(), "offset": list(self.offset)}, f, indent=2)
        print(f"[calibration] saved -> {path}  (offset={self.offset})")

    @classmethod
    def load(cls, path: str = CALIBRATION_FILE) -> "Calibration":
        with open(path) as f:
            data = json.load(f)
        H = np.array(data["H"], dtype=np.float64)
        offset = tuple(data.get("offset", (0.0, 0.0)))  # older files have no offset
        print(f"[calibration] loaded from {path}  (offset={offset})")
        return cls(H, offset)

    @classmethod
    def load_or_run(cls, cap: cv2.VideoCapture, force: bool = False) -> "Calibration":
        if not force and os.path.exists(CALIBRATION_FILE):
            return cls.load()
        return cls._run_interactive(cap)

    @classmethod
    def _run_interactive(cls, cap: cv2.VideoCapture) -> "Calibration":
        points: list[tuple[int, int]] = []
        clone: np.ndarray | None = None

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                label = _CORNER_LABELS[len(points) - 1]
                print(f"[calibration] {label}: ({x}, {y})")

        cv2.destroyAllWindows()    # clear any stale windows (e.g. from find_cam.py)
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WINDOW, 1024, 768)
        cv2.setMouseCallback(_WINDOW, _on_mouse)

        print("[calibration] Click 4 corners in order: TL -> TR -> BR -> BL")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            clone = frame.copy()

            for i, (px, py) in enumerate(points):
                cv2.circle(clone, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(
                    clone,
                    _CORNER_LABELS[i],
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            remaining = 4 - len(points)
            if remaining > 0:
                msg = f"Click {_CORNER_LABELS[len(points)]}  ({remaining} remaining)"
            else:
                msg = "Press ENTER to confirm, R to redo"
            cv2.putText(
                clone, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2
            )

            cv2.imshow(_WINDOW, clone)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                points.clear()
                print("[calibration] Points reset.")
            elif key == ord("q"):
                cv2.destroyWindow(_WINDOW)
                raise RuntimeError("Calibration aborted by user.")
            elif key == 13 and len(points) == 4:  # Enter
                break

        cv2.destroyWindow(_WINDOW)

        src = np.array(points, dtype=np.float32)
        H, _ = cv2.findHomography(src, _GAME_CORNERS)
        cal = cls(H)
        cal.save()
        return cal
