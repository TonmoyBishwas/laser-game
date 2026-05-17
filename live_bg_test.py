"""
Live background-color detection leaderboard.

Cycles the projector through 10 solid-colour fullscreen backgrounds while a
YOLO model runs on the live DroidCam feed. For each colour, measures:

  - hits/sec   : how many frames per second got a laser detection
  - mean_conf  : mean confidence over detections (0..1)
  - miss%      : % of 1-second windows in the active phase with zero detections
  - fp/sec     : false positives per second during a 2 s "quiet" phase at the
                 start of each colour (you DON'T point the laser at the screen)

Run from the project root (so the venv + model paths resolve):

  venv\\Scripts\\python.exe live_bg_test.py
  venv\\Scripts\\python.exe live_bg_test.py --device 0 --quiet 2 --active 15

During each colour:
  - First, a 2 s QUIET phase  -> point the laser OFF the screen (floor, ceiling).
  - Then a 15 s ACTIVE phase  -> sweep the laser around the screen as in Phase 2.

Keys:
  SPACE   skip the current colour
  ESC     abort the run (still saves results for colours completed so far)

Results are saved to bg_test_results/ as JSON + a readable .txt leaderboard.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import pygame
from ultralytics import YOLO

from game import config as cfg


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


def drain_camera(cap: cv2.VideoCapture, n: int = 3) -> None:
    for _ in range(n):
        cap.read()


def overlay_text(screen, font, small, rgb, color_idx, color_name,
                 phase, secs_left, live):
    """Small contrasting panel in top-left so the camera barely sees it."""
    bg_luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    text_color = (10, 10, 10) if bg_luma > 128 else (240, 240, 240)
    panel_color = (255, 255, 255, 90) if bg_luma <= 128 else (0, 0, 0, 90)

    panel = pygame.Surface((620, 130), pygame.SRCALPHA)
    panel.fill(panel_color)
    screen.blit(panel, (12, 12))

    title = font.render(f"{color_idx + 1}/10  {color_name}", True, text_color)
    line2 = small.render(f"[{phase}]  {secs_left:5.1f}s left", True, text_color)
    line3 = small.render(live, True, text_color)
    hint = small.render(
        "QUIET: laser OFF screen.  ACTIVE: sweep laser.  SPACE=skip  ESC=abort",
        True, text_color,
    )
    screen.blit(title, (22, 16))
    screen.blit(line2, (22, 56))
    screen.blit(line3, (22, 80))
    screen.blit(hint,  (22, 104))


def run_phase(cap, model, screen, font, small, rgb, idx, name,
              phase, duration, conf, imgsz):
    """Run a single phase (QUIET or ACTIVE) for `duration` s.

    Returns (total_frames, total_detections, sum_conf, window_hits_list).
    window_hits_list contains, per 1-second wall window, the number of
    detections seen — useful for the miss% calc on the ACTIVE phase.
    """
    total_frames = 0
    total_dets = 0
    sum_conf = 0.0
    windows = []
    win_start = time.time()
    win_dets = 0

    t0 = time.time()
    skipped = False
    aborted = False
    clock = pygame.time.Clock()

    while True:
        elapsed = time.time() - t0
        remaining = duration - elapsed
        if remaining <= 0:
            break

        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    aborted = True
                elif e.key == pygame.K_SPACE:
                    skipped = True
        if aborted or skipped:
            break

        drain_camera(cap, 2)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        result = model.predict(frame, conf=conf, imgsz=imgsz,
                               device="cpu", verbose=False)[0]
        n = len(result.boxes)
        total_frames += 1
        if n > 0:
            total_dets += n
            sum_conf += float(result.boxes.conf.sum().item())
            win_dets += n

        # roll 1-second windows
        now = time.time()
        if now - win_start >= 1.0:
            windows.append(win_dets)
            win_dets = 0
            win_start = now

        hits_so_far = total_dets / max(elapsed, 1e-3)
        mean_conf_so_far = (sum_conf / total_dets) if total_dets > 0 else 0.0
        live = (
            f"frames={total_frames}  dets={total_dets}  "
            f"hits/s={hits_so_far:.1f}  conf={mean_conf_so_far:.2f}"
        )
        screen.fill(rgb)
        overlay_text(screen, font, small, rgb, idx, name,
                     phase, remaining, live)
        pygame.display.flip()
        clock.tick(60)

    # close partial window
    if time.time() - win_start > 0.2:
        windows.append(win_dets)

    return total_frames, total_dets, sum_conf, windows, skipped, aborted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--quiet", type=float, default=2.0,
                   help="seconds of quiet phase (laser OFF) per colour")
    p.add_argument("--active", type=float, default=15.0,
                   help="seconds of active phase (laser sweeping) per colour")
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO confidence threshold (default 0.20)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="YOLO input size (default 320 for CPU speed)")
    p.add_argument("--windowed", action="store_true",
                   help="windowed mode for desk testing")
    p.add_argument("--outdir", type=str, default="bg_test_results")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[bgtest] opening camera {args.device}...")
    cap = open_camera(args.device)
    for _ in range(10):
        cap.read()
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[bgtest] camera fps={cam_fps:.1f} size={cam_w}x{cam_h}")

    print(f"[bgtest] loading YOLO from {cfg.MODEL_PATH}")
    model = YOLO(cfg.MODEL_PATH)
    model.to("cpu")
    print(f"[bgtest] using device=cpu conf={args.conf} imgsz={args.imgsz}")

    pygame.init()
    flags = pygame.FULLSCREEN if not args.windowed else 0
    screen = pygame.display.set_mode((0, 0) if not args.windowed else (1280, 720), flags)
    pygame.display.set_caption("Live BG Test")
    pygame.mouse.set_visible(False)
    font  = pygame.font.SysFont("monospace", 32, bold=True)
    small = pygame.font.SysFont("monospace", 20)

    rows = []
    aborted = False
    try:
        for idx, (name, rgb) in enumerate(COLORS):
            print(f"[bgtest] {idx + 1}/10  {name}  rgb={rgb}")

            # 0.5 s of just showing the colour so the camera adapts
            screen.fill(rgb)
            pygame.display.flip()
            t_show = time.time()
            while time.time() - t_show < 0.5:
                pygame.event.pump()
                screen.fill(rgb)
                pygame.display.flip()

            # QUIET phase (laser OFF) — counts false positives
            qf, qd, qc, _, qskip, qabort = run_phase(
                cap, model, screen, font, small, rgb, idx, name,
                "QUIET", args.quiet, args.conf, args.imgsz)
            if qabort:
                aborted = True
                break
            if qskip:
                # treat skip during quiet as "skip the whole colour"
                rows.append({
                    "name": name, "rgb": list(rgb),
                    "skipped": True,
                    "quiet": {"frames": qf, "detections": qd},
                    "active": None,
                })
                continue

            # ACTIVE phase (laser ON) — measures hits, conf, miss%
            af, ad, ac, windows, askip, aabort = run_phase(
                cap, model, screen, font, small, rgb, idx, name,
                "ACTIVE", args.active, args.conf, args.imgsz)
            if aabort:
                aborted = True
                # still record what we measured
            quiet_secs = max(args.quiet, 1e-3)
            active_secs = max(args.active, 1e-3)
            hits_per_sec = ad / active_secs
            mean_conf = (ac / ad) if ad > 0 else 0.0
            miss_pct = (
                100.0 * sum(1 for w in windows if w == 0) / max(len(windows), 1)
            )
            fp_per_sec = qd / quiet_secs

            row = {
                "name": name,
                "rgb": list(rgb),
                "hits_per_sec": round(hits_per_sec, 2),
                "mean_conf": round(mean_conf, 3),
                "miss_pct": round(miss_pct, 1),
                "fp_per_sec": round(fp_per_sec, 2),
                "active_frames": af,
                "active_detections": ad,
                "quiet_frames": qf,
                "quiet_detections": qd,
                "skipped": askip,
            }
            rows.append(row)
            print(f"[bgtest]   hits/s={hits_per_sec:.1f}  conf={mean_conf:.2f}  "
                  f"miss%={miss_pct:.1f}  fp/s={fp_per_sec:.2f}  "
                  f"{'(skipped active)' if askip else ''}")

            if aborted:
                break
    finally:
        cap.release()
        pygame.quit()

    # ── save results ─────────────────────────────────────────────────────────
    stamp = time.strftime("%Y%m%d_%H%M%S")
    measured = [r for r in rows if r.get("active_frames")]
    ranked = sorted(measured, key=lambda r: r["hits_per_sec"], reverse=True)

    payload = {
        "timestamp": stamp,
        "aborted": aborted,
        "params": {
            "device": args.device,
            "quiet_secs": args.quiet,
            "active_secs": args.active,
            "yolo_conf": args.conf,
            "yolo_imgsz": args.imgsz,
            "model_path": str(cfg.MODEL_PATH),
        },
        "camera": {"fps": cam_fps, "width": cam_w, "height": cam_h},
        "rows": rows,
        "leaderboard": [r["name"] for r in ranked],
    }

    json_path = outdir / f"results_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    (outdir / "latest.json").write_text(json.dumps(payload, indent=2))

    txt_lines = []
    txt_lines.append(f"Live BG test  -  {stamp}")
    txt_lines.append(f"Model: {cfg.MODEL_PATH}")
    txt_lines.append(f"YOLO conf={args.conf}  imgsz={args.imgsz}  "
                     f"quiet={args.quiet}s  active={args.active}s")
    if aborted:
        txt_lines.append("STATUS: ABORTED partway through.")
    txt_lines.append("")
    txt_lines.append(f"{'rank':<5}{'colour':<14}{'hits/s':>9}"
                     f"{'mean_conf':>11}{'miss%':>8}{'fp/s':>8}")
    txt_lines.append("-" * 55)
    for i, r in enumerate(ranked, start=1):
        txt_lines.append(
            f"{i:<5}{r['name']:<14}{r['hits_per_sec']:>9.1f}"
            f"{r['mean_conf']:>11.2f}{r['miss_pct']:>8.1f}{r['fp_per_sec']:>8.2f}"
        )
    if not ranked:
        txt_lines.append("(no colours measured)")
    txt_lines.append("")
    if ranked:
        winner = ranked[0]
        txt_lines.append(f"Best background: {winner['name']}  rgb={tuple(winner['rgb'])}")
        txt_lines.append(f"  -> set BG_COLOR = {tuple(winner['rgb'])} in game/game_loop.py")
    txt = "\n".join(txt_lines)

    txt_path = outdir / f"leaderboard_{stamp}.txt"
    txt_path.write_text(txt, encoding="utf-8")
    (outdir / "latest.txt").write_text(txt, encoding="utf-8")

    print()
    print(txt)
    print()
    print(f"[bgtest] saved {json_path}")
    print(f"[bgtest] saved {txt_path}")
    print(f"[bgtest] also wrote latest.json / latest.txt for convenience.")


if __name__ == "__main__":
    main()
