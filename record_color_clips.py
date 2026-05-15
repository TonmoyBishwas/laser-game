"""
Record per-color laser footage for detection analysis.

Displays each of 10 EIZO-style solid colors fullscreen one at a time, records
the camera in parallel into per-color .mp4 files. Aim the laser at the screen
throughout — try the center, corners, edges, fast moves, slow holds.

After recording, tune detection against each color independently:
  .\venv\Scripts\python.exe tune_detect.py --source clips/01_black.mp4
  .\venv\Scripts\python.exe tune_detect.py --source clips/05_blue.mp4
  ...
  → you'll see exactly which background colors break the detector.

Usage
-----
  .\venv\Scripts\python.exe record_color_clips.py                 # 30s/color (5 min total)
  .\venv\Scripts\python.exe record_color_clips.py --duration 60   # 60s/color (10 min total)
  .\venv\Scripts\python.exe record_color_clips.py --device 1      # different camera
  .\venv\Scripts\python.exe record_color_clips.py --outdir myrun  # different folder

Keys during a clip:
  SPACE    skip the rest of this color, go to next
  ESC      abort the whole session
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import pygame


# ── 10 colors: name + RGB ──────────────────────────────────────────────────────
COLORS = [
    ("01_black",   (  0,   0,   0)),
    ("02_white",   (255, 255, 255)),
    ("03_red",     (255,   0,   0)),
    ("04_green",   (  0, 255,   0)),
    ("05_blue",    (  0,   0, 255)),
    ("06_cyan",    (  0, 255, 255)),
    ("07_magenta", (255,   0, 255)),
    ("08_yellow",  (255, 255,   0)),
    ("09_gray50",  (128, 128, 128)),
    ("10_gray25",  ( 64,  64,  64)),
]


# ── Camera open (same approach as the game) ────────────────────────────────────

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Camera {index} opened but cannot read frames")
    return cap


# ── Background recording thread ────────────────────────────────────────────────

class RecordingThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, out_path: Path, fps: float,
                 size: tuple[int, int], stop_event: threading.Event):
        super().__init__(name=f"Rec-{out_path.name}", daemon=True)
        self.cap = cap
        self.out_path = out_path
        self.fps = fps
        self.size = size
        self.stop_event = stop_event
        self.frames_written = 0

    def run(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, self.size)
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            writer.write(frame)
            self.frames_written += 1
        writer.release()


# ── Display helpers ────────────────────────────────────────────────────────────

def draw_overlay(screen: pygame.Surface, font: pygame.font.Font,
                 small: pygame.font.Font,
                 color_idx: int, color_name: str, color_rgb: tuple[int, int, int],
                 seconds_left: float, total_secs: float,
                 frames_written: int) -> None:
    """Tiny overlay in a corner — chosen to minimise camera contamination."""
    # Choose overlay text color that contrasts with the background
    bg_luma = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
    text_color = (10, 10, 10) if bg_luma > 128 else (240, 240, 240)
    panel_color = (255, 255, 255, 80) if bg_luma <= 128 else (0, 0, 0, 80)

    panel_w, panel_h = 540, 100
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(panel_color)
    screen.blit(panel, (12, 12))

    title = font.render(f"{color_idx + 1}/10  {color_name}", True, text_color)
    sub   = small.render(
        f"{seconds_left:5.1f}s left / {total_secs:.0f}s   frames={frames_written}",
        True, text_color,
    )
    hint  = small.render(
        "Aim the laser around the screen (centre, corners, slow + fast). SPACE=skip  ESC=abort",
        True, text_color,
    )
    screen.blit(title, (22, 16))
    screen.blit(sub,   (22, 56))
    screen.blit(hint,  (22, 82))


# ── Main run ──────────────────────────────────────────────────────────────────

def run(device: int, duration: float, outdir: str, fullscreen: bool, warmup: float) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[record] opening camera {device}...")
    cap = open_camera(device)
    # Measure camera params
    for _ in range(10):
        cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[record] camera FPS={fps:.1f}  size={w}x{h}")

    pygame.init()
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((0, 0) if fullscreen else (1280, 720), flags)
    pygame.display.set_caption("Color background recorder")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 32, bold=True)
    small = pygame.font.SysFont("monospace", 22)

    metadata = {
        "duration_per_color_secs": duration,
        "camera_fps": fps,
        "camera_size": [w, h],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup_secs": warmup,
        "colors": [],
    }

    aborted = False
    try:
        for idx, (name, rgb) in enumerate(COLORS):
            print(f"[record] {idx + 1}/10  {name}  rgb={rgb}  for {duration}s")

            # Fill the whole screen and let camera see it for `warmup` seconds
            # before recording starts (so the recording is clean from frame 0).
            screen.fill(rgb)
            pygame.display.flip()
            t_warm = time.time()
            while time.time() - t_warm < warmup:
                for e in pygame.event.get():
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        aborted = True
                        break
                if aborted:
                    break
                # Keep the screen filled in case something else paints over
                screen.fill(rgb)
                pygame.display.flip()
                clock.tick(60)
            if aborted:
                break

            # Start the recording thread for this color
            file_path = out_path / f"{name}.mp4"
            stop_event = threading.Event()
            rec = RecordingThread(cap, file_path, fps, (w, h), stop_event)
            rec.start()

            t0 = time.time()
            skipped = False
            while True:
                elapsed = time.time() - t0
                remaining = duration - elapsed
                if remaining <= 0:
                    break

                # Event handling
                for e in pygame.event.get():
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_ESCAPE:
                            aborted = True
                        elif e.key == pygame.K_SPACE:
                            skipped = True
                if aborted or skipped:
                    break

                # Render
                screen.fill(rgb)
                draw_overlay(screen, font, small, idx, name, rgb,
                             remaining, duration, rec.frames_written)
                pygame.display.flip()
                clock.tick(60)

            stop_event.set()
            rec.join(timeout=2)
            secs = time.time() - t0
            print(f"[record]   wrote {rec.frames_written} frames "
                  f"({rec.frames_written / max(secs, 1e-3):.1f} fps actual) "
                  f"-> {file_path}  {'(SKIPPED)' if skipped else ''}")
            metadata["colors"].append({
                "name": name,
                "rgb": list(rgb),
                "file": file_path.name,
                "frames": rec.frames_written,
                "actual_secs": secs,
                "skipped": skipped,
            })

            if aborted:
                break
    finally:
        cap.release()
        pygame.quit()

    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["aborted"] = aborted
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print()
    print(f"[record] {'ABORTED' if aborted else 'DONE'}. metadata -> {out_path / 'metadata.json'}")
    print(f"[record] now tune any clip with:")
    print(f"           .\\venv\\Scripts\\python.exe tune_detect.py --source {out_path}/<name>.mp4")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=30.0,
                   help="seconds per color (default 30 -> 5 min total)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--outdir", type=str, default="clips")
    p.add_argument("--windowed", action="store_true",
                   help="run in a window instead of fullscreen (testing only)")
    p.add_argument("--warmup", type=float, default=1.0,
                   help="seconds to display each color before recording starts")
    args = p.parse_args()
    run(device=args.device, duration=args.duration, outdir=args.outdir,
        fullscreen=not args.windowed, warmup=args.warmup)


if __name__ == "__main__":
    main()
