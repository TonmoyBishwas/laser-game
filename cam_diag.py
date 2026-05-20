"""
Camera diagnostic for the DroidCam green-feed problem.

Opens one camera, reports the pixel format OpenCV actually negotiated,
checks whether frames are actually CHANGING (live) or frozen, tries a
couple of fixes, and saves sample frames to disk so we can see what the
"green" actually is.

Run on the laptop that has DroidCam connected:

  venv\\Scripts\\python.exe cam_diag.py --device 1 --backend MSMF
  venv\\Scripts\\python.exe cam_diag.py --device 1 --backend AUTO

Paste the console output back, and open the PNGs in cam_diag_out\\.
"""

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


BACKENDS = {"AUTO": cv2.CAP_ANY, "DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}


def fourcc_to_str(val: float) -> str:
    code = int(val)
    if code <= 0:
        return f"({code})"
    try:
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return f"({code})"


def describe(cap: cv2.VideoCapture) -> None:
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    convert_rgb = cap.get(cv2.CAP_PROP_CONVERT_RGB)
    print(f"  negotiated: {int(w)}x{int(h)} @ {fps:.1f}fps  "
          f"fourcc='{fourcc}'  convert_rgb={convert_rgb}")


def sample_means(cap: cv2.VideoCapture, n: int = 60) -> list:
    """Read n frames over time and return per-frame mean + a few saved frames."""
    means = []
    saved = {}
    save_at = {0, n // 2, n - 1}
    for i in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            means.append(None)
            continue
        m = float(np.mean(frame))
        means.append(m)
        if i in save_at:
            saved[i] = frame.copy()
        time.sleep(0.03)
    return means, saved


def analyse(label: str, means: list) -> None:
    valid = [m for m in means if m is not None]
    if not valid:
        print(f"  [{label}] NO frames read at all.")
        return
    lo, hi = min(valid), max(valid)
    spread = hi - lo
    print(f"  [{label}] {len(valid)}/{len(means)} frames read.  "
          f"mean range = {lo:.2f} .. {hi:.2f}  (spread={spread:.2f})")
    if spread < 1.0:
        print(f"  [{label}] VERDICT: frames are FROZEN/static (spread<1). "
              f"The stream is NOT flowing into OpenCV — this is a "
              f"connection/format negotiation problem, not a colour bug.")
    else:
        print(f"  [{label}] VERDICT: frames ARE changing (live). If they "
              f"look green, it's a colour DECODE problem we can fix.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--backend", choices=list(BACKENDS), default="MSMF")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--out", type=str, default="cam_diag_out")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    backend = BACKENDS[args.backend]
    print(f"\n=== opening index={args.device} backend={args.backend} ===")
    cap = cv2.VideoCapture(args.device, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("  cannot open. Done.")
        return

    # warmup
    for _ in range(10):
        cap.read()

    print("\n[1] default settings:")
    describe(cap)
    means, saved = sample_means(cap, args.frames)
    analyse("default", means)
    for i, frame in saved.items():
        path = out / f"{args.backend}_dev{args.device}_default_{i:03d}.png"
        cv2.imwrite(str(path), frame)
        print(f"      saved {path}")

    # try forcing CONVERT_RGB on
    print("\n[2] forcing CAP_PROP_CONVERT_RGB = 1:")
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)
    describe(cap)
    means2, saved2 = sample_means(cap, args.frames // 2)
    analyse("convert_rgb=1", means2)
    for i, frame in saved2.items():
        path = out / f"{args.backend}_dev{args.device}_convrgb_{i:03d}.png"
        cv2.imwrite(str(path), frame)
        print(f"      saved {path}")

    # try forcing MJPG
    print("\n[3] forcing FOURCC = MJPG:")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    describe(cap)
    means3, saved3 = sample_means(cap, args.frames // 2)
    analyse("mjpg", means3)
    for i, frame in saved3.items():
        path = out / f"{args.backend}_dev{args.device}_mjpg_{i:03d}.png"
        cv2.imwrite(str(path), frame)
        print(f"      saved {path}")

    cap.release()
    print(f"\nDone. Open the PNGs in {out}\\ and paste the console output.")


if __name__ == "__main__":
    main()
