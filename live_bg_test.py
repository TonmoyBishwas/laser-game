"""
Live background-color detection leaderboard.

Cycles the projector through 30 solid-colour fullscreen backgrounds while a
YOLO model runs on the live DroidCam feed. For each colour, measures:

  - hits/sec   : how many frames per second got a laser detection
  - mean_conf  : mean confidence over detections (0..1)
  - miss%      : % of 1-second windows in the active phase with zero detections
  - fp/sec     : false positives per second during the optional QUIET phase
                 (only measured if you pass --quiet N)

Run from the project root (so the venv + model paths resolve):

  venv\\Scripts\\python.exe live_bg_test.py
  venv\\Scripts\\python.exe live_bg_test.py --start 10            # darker only
  venv\\Scripts\\python.exe live_bg_test.py --quiet 2 --active 15 # with FP

By default each colour just runs an ACTIVE phase: sweep the laser around the
screen the entire 15 s — centre, corners, edges, fast + slow.

Keys:
  SPACE   skip the current colour
  ESC     abort the run (still saves results for colours completed so far)

Results are saved to bg_test_results/ as JSON + a readable .txt leaderboard.
"""

import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import pygame
from ultralytics import YOLO

from game import config as cfg
from bg_palette import COLORS


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


def overlay_text(screen, font, small, rgb, color_idx, color_total, color_name,
                 phase, secs_label, live, hint_text=None):
    """Small contrasting panel in top-left so the camera barely sees it."""
    bg_luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    text_color = (10, 10, 10) if bg_luma > 128 else (240, 240, 240)
    panel_color = (255, 255, 255, 90) if bg_luma <= 128 else (0, 0, 0, 90)

    panel = pygame.Surface((620, 130), pygame.SRCALPHA)
    panel.fill(panel_color)
    screen.blit(panel, (12, 12))

    title = font.render(f"{color_idx + 1}/{color_total}  {color_name}", True, text_color)
    line2 = small.render(f"[{phase}]  {secs_label}", True, text_color)
    line3 = small.render(live, True, text_color)
    hint = small.render(
        hint_text or "QUIET: laser OFF screen.  ACTIVE: sweep laser.  SPACE=skip  ESC=abort",
        True, text_color,
    )
    screen.blit(title, (22, 16))
    screen.blit(line2, (22, 56))
    screen.blit(line3, (22, 80))
    screen.blit(hint,  (22, 104))


def draw_history_strip(screen, history, x, y, cell=22, gap=2):
    """Row of cells: bright green if frame had detection, dark red if not.

    Newest on the right. Lets the player see continuous vs flaky vs lost
    detection patterns at a glance.
    """
    n_cells = history.maxlen
    total_w = n_cells * (cell + gap) - gap
    # background tray with subtle border so the strip is visible on any bg
    tray = pygame.Surface((total_w + 8, cell + 8), pygame.SRCALPHA)
    tray.fill((0, 0, 0, 110))
    pygame.draw.rect(tray, (255, 255, 255, 120),
                     tray.get_rect(), width=1)
    screen.blit(tray, (x - 4, y - 4))

    # pad with False on the left if not full yet
    padding = n_cells - len(history)
    cells = [False] * padding + list(history)
    for i, hit in enumerate(cells):
        cx = x + i * (cell + gap)
        if hit:
            pygame.draw.rect(screen, (0, 230, 0), (cx, y, cell, cell))
        else:
            pygame.draw.rect(screen, (140, 0, 0), (cx, y, cell, cell))


def draw_big_indicator(screen, since_det):
    """Large pulsing circle in top-right.

    Bright lime for ~0.1 s after a detection, fading to dark grey over
    ~0.6 s. Always has a white outline so it's visible on any background.
    """
    size = 180
    cx = screen.get_width() - size // 2 - 30
    cy = size // 2 + 30

    if since_det < 0.1:
        ratio = 1.0
    elif since_det < 0.6:
        ratio = 1.0 - (since_det - 0.1) / 0.5
    else:
        ratio = 0.0

    if ratio > 0.05:
        color = (int(20 + 30 * ratio),
                 int(80 + 175 * ratio),
                 int(20 + 30 * ratio))
    else:
        color = (50, 50, 50)

    pygame.draw.circle(screen, color, (cx, cy), size // 2)
    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), size // 2, width=4)
    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), size // 4, width=2)


def _resolve_hold_start(value):
    """Return (start_idx, start_rgb, label).

    start_idx == -1 means custom RGB (not in palette). Other values are
    valid palette indices. Returns None if `value` cannot be parsed.
    """
    if value in ("browse", ""):
        name, rgb = COLORS[0]
        return 0, rgb, name
    for i, (name, rgb) in enumerate(COLORS):
        if value == name:
            return i, rgb, name
    if "," in value:
        try:
            parts = [int(x.strip()) for x in value.split(",")]
            if len(parts) == 3 and all(0 <= p <= 255 for p in parts):
                return -1, tuple(parts), f"custom({parts[0]},{parts[1]},{parts[2]})"
        except ValueError:
            pass
    return None


def _run_hold(cap, model, screen, font, small,
              start_idx, start_rgb, start_label, conf, imgsz):
    """Stay on a chosen colour and let the player switch via arrow keys.

    Shows a large pulsing detection indicator (top-right) and a 30-cell
    rolling history strip (top-left, under the text panel). Stats reset
    every time the colour changes.
    """
    current_idx = start_idx
    rgb = start_rgb
    label = start_label

    history = collections.deque(maxlen=30)
    t_start = time.time()
    frames = 0
    dets = 0
    sum_conf = 0.0
    last_det_time = -10.0

    def reset_stats():
        nonlocal t_start, frames, dets, sum_conf, last_det_time
        t_start = time.time()
        frames = 0
        dets = 0
        sum_conf = 0.0
        last_det_time = -10.0
        history.clear()

    def shift_idx(delta):
        nonlocal current_idx, rgb, label
        if current_idx == -1:
            current_idx = 0 if delta > 0 else len(COLORS) - 1
        else:
            current_idx = (current_idx + delta) % len(COLORS)
        label, rgb = COLORS[current_idx][0], COLORS[current_idx][1]
        reset_stats()

    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    print(f"[bgtest] HOLD exited at {label}. "
                          f"final: frames={frames} dets={dets} "
                          f"hits/s={dets / max(time.time() - t_start, 1e-3):.1f}")
                    return
                if e.key in (pygame.K_RIGHT, pygame.K_DOWN):
                    shift_idx(+1)
                elif e.key in (pygame.K_LEFT, pygame.K_UP):
                    shift_idx(-1)
                elif e.key == pygame.K_PAGEUP:
                    shift_idx(+5)
                elif e.key == pygame.K_PAGEDOWN:
                    shift_idx(-5)
                elif e.key == pygame.K_HOME:
                    shift_idx(-current_idx if current_idx >= 0 else +1)
                elif e.key == pygame.K_END:
                    shift_idx((len(COLORS) - 1 - current_idx)
                              if current_idx >= 0 else -1)
                elif e.key == pygame.K_r:
                    reset_stats()

        drain_camera(cap, 2)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        result = model.predict(frame, conf=conf, imgsz=imgsz,
                               device="cpu", verbose=False)[0]
        n = len(result.boxes)
        frames += 1
        detected = n > 0
        history.append(detected)
        if detected:
            dets += n
            sum_conf += float(result.boxes.conf.sum().item())
            last_det_time = time.time()

        elapsed = max(time.time() - t_start, 1e-3)
        hits_per_sec = dets / elapsed
        mean_conf = (sum_conf / dets) if dets > 0 else 0.0
        rolling_pct = 100.0 * sum(history) / len(history) if history else 0.0
        live = (
            f"frames={frames}  dets={dets}  hits/s={hits_per_sec:.1f}  "
            f"conf={mean_conf:.2f}  last30={rolling_pct:.0f}%"
        )

        idx_label = f"{current_idx + 1}/{len(COLORS)}" if current_idx >= 0 else "custom"
        nav_hint = ("<-/-> change colour  PgUp/Dn jump 5  HOME/END ends  "
                    "R reset  ESC exit")

        screen.fill(rgb)
        # show palette index in the title (use color_idx/total for the existing renderer)
        ci = current_idx if current_idx >= 0 else 0
        ct = len(COLORS)
        overlay_text(screen, font, small, rgb, ci, ct, label,
                     "LIVE", f"{elapsed:.0f}s elapsed", live,
                     hint_text=nav_hint)
        draw_history_strip(screen, history, x=22, y=160)
        draw_big_indicator(screen, time.time() - last_det_time)

        pygame.display.flip()
        clock.tick(60)


def run_phase(cap, model, screen, font, small, rgb, idx, total, name,
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
        overlay_text(screen, font, small, rgb, idx, total, name,
                     phase, f"{remaining:5.1f}s left", live)
        pygame.display.flip()
        clock.tick(60)

    # close partial window
    if time.time() - win_start > 0.2:
        windows.append(win_dets)

    return total_frames, total_dets, sum_conf, windows, skipped, aborted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--quiet", type=float, default=0.0,
                   help="seconds of quiet phase (laser OFF) for false-positive "
                        "measurement. Default 0 = skip. Pass e.g. --quiet 2 to "
                        "enable.")
    p.add_argument("--active", type=float, default=15.0,
                   help="seconds of active phase (laser sweeping) per colour")
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO confidence threshold (default 0.20)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="YOLO input size (default 320 for CPU speed)")
    p.add_argument("--windowed", action="store_true",
                   help="windowed mode for desk testing")
    p.add_argument("--outdir", type=str, default="bg_test_results")
    p.add_argument("--limit", type=int, default=0,
                   help="only test the first N colours (0 = all 30)")
    p.add_argument("--start", type=int, default=0,
                   help="skip the first N colours (e.g. --start 10 to only "
                        "test the darker additions)")
    p.add_argument("--hold", type=str, nargs="?", const="browse", default=None,
                   help="fullscreen one colour and run detection forever. "
                        "Use plain '--hold' to start at colour 1 and arrow "
                        "through, '--hold 15_dark_blue' to start there, or "
                        "'--hold 0,0,64' for a custom RGB. ESC to exit.")
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

    # ── HOLD MODE: stay on one colour, arrow-key through palette ─────────────
    if args.hold is not None:
        resolved = _resolve_hold_start(args.hold)
        if resolved is None:
            print(f"[bgtest] unknown --hold value '{args.hold}'. "
                  f"Use a name from bg_palette.py, 'browse', or an R,G,B triple.")
            cap.release()
            pygame.quit()
            return
        start_idx, start_rgb, start_label = resolved
        print(f"[bgtest] HOLD mode on {start_label} rgb={start_rgb}. "
              f"<-/-> to change colour, ESC to exit.")
        _run_hold(cap, model, screen, font, small,
                  start_idx, start_rgb, start_label,
                  args.conf, args.imgsz)
        cap.release()
        pygame.quit()
        return

    selected = COLORS[args.start:]
    if args.limit > 0:
        selected = selected[:args.limit]
    print(f"[bgtest] testing {len(selected)} colours "
          f"(from index {args.start})")

    rows = []
    aborted = False
    total = len(selected)
    try:
        for idx, (name, rgb) in enumerate(selected):
            print(f"[bgtest] {idx + 1}/{total}  {name}  rgb={rgb}")

            # 0.5 s of just showing the colour so the camera adapts
            screen.fill(rgb)
            pygame.display.flip()
            t_show = time.time()
            while time.time() - t_show < 0.5:
                pygame.event.pump()
                screen.fill(rgb)
                pygame.display.flip()

            # QUIET phase (laser OFF) — only if --quiet > 0
            qf, qd = 0, 0
            if args.quiet > 0:
                qf, qd, _qc, _, qskip, qabort = run_phase(
                    cap, model, screen, font, small, rgb, idx, total, name,
                    "QUIET", args.quiet, args.conf, args.imgsz)
                if qabort:
                    aborted = True
                    break
                if qskip:
                    rows.append({
                        "name": name, "rgb": list(rgb),
                        "skipped": True,
                        "quiet": {"frames": qf, "detections": qd},
                        "active": None,
                    })
                    continue

            # ACTIVE phase (laser ON) — measures hits, conf, miss%
            af, ad, ac, windows, askip, aabort = run_phase(
                cap, model, screen, font, small, rgb, idx, total, name,
                "ACTIVE", args.active, args.conf, args.imgsz)
            if aabort:
                aborted = True
                # still record what we measured
            active_secs = max(args.active, 1e-3)
            hits_per_sec = ad / active_secs
            mean_conf = (ac / ad) if ad > 0 else 0.0
            miss_pct = (
                100.0 * sum(1 for w in windows if w == 0) / max(len(windows), 1)
            )
            fp_per_sec = (qd / args.quiet) if args.quiet > 0 else 0.0

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
