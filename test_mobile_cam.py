r"""
Mobile camera app tester  --  "is my phone app actually streaming to the PC?"

This is the IriunCam-vs-DroidCam sanity check for the custom Android app
(see MOBILE_CAM_APP_NOTES.md). The app serves MJPEG over HTTP at
  http://<phone-ip>:8080/video      (Wi-Fi)
  http://localhost:8080/video       (USB, after: adb forward tcp:8080 tcp:8080)

You do NOT need the projector, the game, calibration, or YOLO to run this.
Just have the phone app streaming and the phone reachable from the PC.

What it proves, in order of increasing confidence:
  1. The URL opens at all                       (app running + reachable?)
  2. OpenCV can DECODE frames                    (right format = MJPEG?)
  3. The feed is LIVE, not a frozen placeholder  (the exact DroidCam failure)
  4. It streams at a usable FPS with low latency (smooth enough for the game?)

The decisive test is #3: a live feed's per-frame mean brightness always
varies; DroidCam's broken virtual-cam handed back a static buffer with mean
spread = 0.00. We measure that spread and give a clear PASS / FAIL verdict.

Usage
-----
  # Wi-Fi (most common at home): use the IP the app shows on its screen
  .\venv\Scripts\python.exe test_mobile_cam.py --url http://192.168.0.42:8080/video

  # USB: first run  adb forward tcp:8080 tcp:8080  then:
  .\venv\Scripts\python.exe test_mobile_cam.py --url http://localhost:8080/video

  # headless (no preview window), e.g. over SSH or quick CI-style check:
  .\venv\Scripts\python.exe test_mobile_cam.py --url ... --no-preview

In the preview window press  q  or  ESC  to stop early.
"""

import argparse
import time

import cv2
import numpy as np


def fourcc_to_str(val: float) -> str:
    code = int(val)
    if code <= 0:
        return "(none)"
    try:
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip()
    except Exception:
        return f"({code})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8080/video",
                   help="MJPEG stream URL the app serves (default: USB/localhost)")
    p.add_argument("--seconds", type=float, default=8.0,
                   help="how long to sample the stream (default: 8s)")
    p.add_argument("--no-preview", action="store_true",
                   help="don't open a window; just print the verdict")
    args = p.parse_args()

    print("=" * 70)
    print(f"  Testing mobile camera stream:  {args.url}")
    print("=" * 70)

    # cv2.VideoCapture reads an MJPEG-over-HTTP URL natively -- no virtual
    # camera, no device index, no DSHOW/MSMF driver layer (that's the whole
    # point of the new app vs DroidCam).
    t_open = time.time()
    cap = cv2.VideoCapture(args.url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the freshest frame
    open_secs = time.time() - t_open

    if not cap.isOpened():
        print("\n  [FAIL] Could not open the stream.")
        print("  Check, in order:")
        print("    - Is the app actually streaming? (open the URL in a browser first)")
        print("    - Wi-Fi: is the IP correct + phone & PC on the SAME network?")
        print("    - USB:  did you run  adb forward tcp:8080 tcp:8080  ?")
        print("    - Is the port right (default 8080) and path /video ?")
        cap.release()
        return

    # --- time-to-first-frame (how long until the PC sees real video) ---
    t0 = time.time()
    ok, frame = None, None
    while time.time() - t0 < 10.0:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
    first_frame_secs = time.time() - t0

    if not ok or frame is None or frame.size == 0:
        print(f"\n  [FAIL] Stream opened but delivered NO decodable frame in 10s.")
        print("  This is the classic broken-feed symptom. Likely causes:")
        print("    - the app is serving the wrong format (must be MJPEG / image/jpeg)")
        print("    - the app is serving a frozen placeholder instead of failing loudly")
        cap.release()
        return

    h, w = frame.shape[:2]
    fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"\n  Opened in {open_secs:.2f}s,  first frame in {first_frame_secs:.2f}s")
    print(f"  Resolution: {w}x{h}   reported format: '{fourcc}'")
    print(f"  Sampling for {args.seconds:.0f}s ...  (press q / ESC in the window to stop)\n")

    # --- sample the stream: count frames, measure FPS, and the LIVE check ---
    means = []
    n_read = n_fail = 0
    win = "Mobile cam test  (q/ESC = quit)"
    if not args.no_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_start = time.time()
    while time.time() - t_start < args.seconds:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            n_fail += 1
            continue
        n_read += 1
        means.append(float(np.mean(frame)))

        if not args.no_preview:
            elapsed = time.time() - t_start
            fps_live = n_read / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, f"{w}x{h}  ~{fps_live:.1f} fps  frame {n_read}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    elapsed = time.time() - t_start
    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    # --- verdict ---
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    if not means:
        print("  [FAIL] No frames read during sampling. Stream stalled after opening.")
        return

    fps = n_read / elapsed if elapsed > 0 else 0.0
    lo, hi = min(means), max(means)
    spread = hi - lo
    print(f"  frames read : {n_read}   (read failures: {n_fail})")
    print(f"  measured fps: {fps:.1f}")
    print(f"  brightness  : mean range {lo:.2f} .. {hi:.2f}   (spread = {spread:.2f})")
    print()

    # LIVE vs FROZEN -- the exact check DroidCam failed (spread was 0.00).
    live = spread >= 1.0
    fast_enough = fps >= 10.0

    if live:
        print("  [PASS] Feed is LIVE  -- per-frame brightness varies, real video is")
        print("         flowing into OpenCV (this is what DroidCam could NOT do).")
    else:
        print("  [FAIL] Feed looks FROZEN/static (brightness spread < 1.0).")
        print("         OpenCV is getting a placeholder, not live video -- same")
        print("         failure mode as DroidCam. Fix the app before integrating:")
        print("         the app must serve real MJPEG frames, never a frozen buffer.")

    if live and not fast_enough:
        print(f"  [WARN] fps is low ({fps:.1f}). It works, but may feel laggy in-game.")
        print("         Try lowering the app's resolution or JPEG quality, prefer")
        print("         USB (adb forward) over Wi-Fi, or move closer to the router.")

    print()
    if live and fast_enough:
        print("  >> SUCCESS: the app streams smoothly and OpenCV reads it cleanly.")
        print("     This is the IriunCam-grade result we wanted. Ready for the game:")
        print(f"     it'll plug in as  cv2.VideoCapture(\"{args.url}\")")
    elif live:
        print("  >> WORKS, but tune for speed (see WARN above) before the live demo.")
    print()


if __name__ == "__main__":
    main()
