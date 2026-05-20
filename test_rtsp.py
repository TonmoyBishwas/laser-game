r"""
RTSP camera access test.

Paste an RTSP URL (from your phone streaming app) and this opens it,
shows the live feed, and tells you whether real video is flowing (the
per-frame brightness changes) or the stream is frozen/unreachable.

This is the camera-access check ONLY. Once this shows live video, the
same URL can be fed to the detection step.

Usage
-----
  venv\Scripts\python.exe test_rtsp.py --url rtsp://192.168.1.50:5540/ch0
  venv\Scripts\python.exe test_rtsp.py            (it will prompt for the URL)

Keys in the window:
  q or ESC   quit

Notes
-----
- RTSP connections can take 5-10 seconds to open. Be patient.
- We force TCP transport (rtsp_transport=tcp) which is far more stable
  than UDP for these phone apps (less tearing / corruption).
"""

import os
# Force RTSP over TCP for stability. MUST be set before cv2 is imported.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import argparse
import time

import cv2
import numpy as np


def open_rtsp(url: str) -> cv2.VideoCapture:
    # FFMPEG is the backend that handles RTSP in OpenCV.
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, default="",
                   help="RTSP URL, e.g. rtsp://192.168.1.50:5540/ch0")
    p.add_argument("--display-width", type=int, default=1280,
                   help="max preview width; big streams are downscaled to fit")
    p.add_argument("--seconds", type=float, default=0,
                   help="auto-stop after N seconds (0 = run until you press q)")
    args = p.parse_args()

    url = args.url.strip()
    if not url:
        try:
            url = input("Paste your RTSP URL and press Enter:\n> ").strip()
        except EOFError:
            url = ""
    if not url:
        print("[rtsp] no URL given. Re-run with --url rtsp://...")
        return

    print(f"[rtsp] opening: {url}")
    print("[rtsp] (RTSP can take 5-10s to connect — be patient)...")
    cap = open_rtsp(url)
    if not cap.isOpened():
        print("[rtsp] FAILED to open the stream. Check that:")
        print("       - the phone app is actively streaming")
        print("       - the URL (IP + port + path) is exactly right")
        print("       - phone and PC are on the same network (for Wi-Fi)")
        print("       - no firewall is blocking the port")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rep_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[rtsp] connected. reported {w}x{h} @ {rep_fps:.1f} fps")

    win = "RTSP test  -  q/ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    means = []
    frames = 0
    fail_reads = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_reads += 1
            if fail_reads > 200:
                print("[rtsp] too many failed reads — stream dropped or never started.")
                break
            # brief wait so we don't spin
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                break
            continue
        fail_reads = 0
        frames += 1
        m = float(np.mean(frame))
        means.append(m)

        elapsed = max(time.time() - t0, 1e-3)
        meas_fps = frames / elapsed
        recent = means[-30:]
        recent_spread = (max(recent) - min(recent)) if len(recent) > 1 else 0.0
        live_flag = "LIVE" if recent_spread >= 1.0 else "STATIC?"

        disp = frame
        if disp.shape[1] > args.display_width:
            scale = args.display_width / disp.shape[1]
            disp = cv2.resize(frame, (int(disp.shape[1] * scale),
                                      int(disp.shape[0] * scale)))

        lines = [
            f"{w}x{h}  measured_fps={meas_fps:.1f}",
            f"frames={frames}  mean={m:.1f}  recent_spread={recent_spread:.1f}",
            f"status: {live_flag}",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 28
            cv2.rectangle(disp, (8, y - 20), (8 + 12 * len(line), y + 8),
                          (0, 0, 0), -1)
            cv2.putText(disp, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (40, 255, 255), 2)

        cv2.imshow(win, disp)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        if args.seconds > 0 and elapsed >= args.seconds:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── verdict ──────────────────────────────────────────────────────────────
    print()
    if frames == 0:
        print("[rtsp] VERDICT: no frames read at all. The URL opened but no "
              "video arrived — wrong path, or the app isn't pushing video.")
        return
    spread = max(means) - min(means)
    avg_fps = frames / max(time.time() - t0, 1e-3)
    print(f"[rtsp] read {frames} frames, avg {avg_fps:.1f} fps, "
          f"brightness range {min(means):.1f}..{max(means):.1f} "
          f"(spread={spread:.1f})")
    if spread < 1.0:
        print("[rtsp] VERDICT: FROZEN — frames never changed. The stream is "
              "static; check the app is actually capturing the camera.")
    else:
        print("[rtsp] VERDICT: LIVE — real video is flowing. This URL is good "
              "to use for detection.")


if __name__ == "__main__":
    main()
