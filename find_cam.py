"""
Find every camera device this PC can see, test which ones return real frames,
and show a live preview so you can confirm which one is DroidCam.

Usage
-----
  .\venv\Scripts\python.exe find_cam.py            # scan + preview best one
  .\venv\Scripts\python.exe find_cam.py --preview 1 --backend MSMF   # force preview

In the preview window, press:
  q  → quit
  n  → try next working camera (cycles through them)
"""

import argparse
import sys
import cv2
import numpy as np


BACKENDS = {
    "AUTO":  cv2.CAP_ANY,
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF":  cv2.CAP_MSMF,
}


def open_cam(index: int, backend: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def test_capture(index: int, backend: int) -> dict:
    cap = open_cam(index, backend)
    result = {
        "index":  index,
        "backend": [k for k, v in BACKENDS.items() if v == backend][0],
        "opened": cap.isOpened(),
        "frame":  False,
        "mean":   None,
        "shape":  None,
    }
    if not cap.isOpened():
        cap.release()
        return result
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
        result["frame"] = True
        result["mean"] = round(float(np.mean(frame)), 1)
        result["shape"] = tuple(frame.shape)
    cap.release()
    return result


def scan() -> list[dict]:
    print("=" * 78)
    print("  Scanning all camera indices on all backends...")
    print("=" * 78)
    results = []
    for bname, bcode in BACKENDS.items():
        for i in range(5):
            r = test_capture(i, bcode)
            results.append(r)
            print(
                f"  index={i}  backend={bname:<5}"
                f"  opened={str(r['opened']):<5}  frame={str(r['frame']):<5}"
                f"  mean={r['mean']}  shape={r['shape']}"
            )
    return results


def preview(index: int, backend: int) -> None:
    """Show live feed so the user can visually confirm which camera this is."""
    cap = open_cam(index, backend)
    if not cap.isOpened():
        print(f"[preview] cannot open index={index} backend={backend}")
        return

    bname = [k for k, v in BACKENDS.items() if v == backend][0]
    win = f"Preview index={index} backend={bname}  (q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"\n[preview] showing index={index} backend={bname}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[preview] read failed")
            break
        # overlay
        mean_v = float(np.mean(frame))
        cv2.putText(
            frame,
            f"index={index} backend={bname} mean={mean_v:.1f} shape={frame.shape}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=int, default=None, help="Force preview of this index")
    parser.add_argument("--backend", choices=list(BACKENDS), default="AUTO", help="Backend for preview")
    args = parser.parse_args()

    if args.preview is not None:
        preview(args.preview, BACKENDS[args.backend])
        return

    results = scan()
    working = [r for r in results if r["frame"] and r["mean"] is not None]

    print("\n" + "=" * 78)
    print("  Working cameras (returns real frames):")
    print("=" * 78)
    if not working:
        print("  NONE. Check DroidCam PC client is running with the phone connected.")
        sys.exit(1)

    for r in working:
        print(f"  index={r['index']}  backend={r['backend']}  mean={r['mean']}  shape={r['shape']}")

    # Cycle preview through every working camera so user can identify DroidCam visually
    print("\n" + "=" * 78)
    print("  Launching preview for each working camera so you can identify DroidCam.")
    print("  Press 'q' in each window to advance to the next.")
    print("=" * 78)
    seen = set()
    for r in working:
        key = (r["index"], r["backend"])
        if key in seen:
            continue
        seen.add(key)
        preview(r["index"], BACKENDS[r["backend"]])


if __name__ == "__main__":
    main()
