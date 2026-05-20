r"""
Flashlight sandbox — tune the spotlight in isolation.

Just the laser->flashlight reveal: no thief, no maze logic, no game-over,
runs forever so you can test and tune the flashlight feel as long as you
like. Reads the SAME camera + YOLO + calibration pipeline the real game
uses, so what you tune here transfers directly.

Why the light was offset from your laser
----------------------------------------
The game draws the spotlight exactly at the homography-mapped point with
no correction. So an offset means the 4-corner CALIBRATION was slightly
off (very easy when you clicked through the infinity-mirror recursion) or
the phone lens has distortion. This sandbox lets you:
  - see the RAW mapped point (red +) vs the CORRECTED point (green +),
  - dial in an X/Y offset to line the spotlight up with your laser,
  - press R to recalibrate carefully if the offset grows toward the edges
    (that pattern means perspective/lens error, which an offset can't fix).

Tunable sliders (drag with the mouse, or select with UP/DOWN and adjust
with LEFT/RIGHT; hold steps with PageUp/PageDown for big jumps):
  radius      spotlight size (px)
  offset_x/y  correction added to the mapped laser point (px)
  smoothing   0 = instant (snappy), higher = smoother but laggier
  fade_secs   how fast the light fades in/out
  stale_secs  how long a detection stays "fresh" (raise to stop flicker
              at low detection FPS)
  falloff     edge softness of the light
  dark_alpha  how black the unlit area is

Keys:
  UP/DOWN        select a slider
  LEFT/RIGHT     adjust selected (fine)
  PageUp/PageDn  adjust selected (coarse)
  mouse drag     set a slider directly
  C              reset offsets to 0
  G              toggle the revealable background grid
  S              save values to flashlight_tuning.json (+ prints config)
  Q / ESC        quit
  (to recalibrate, restart with --recalibrate)

Usage
-----
  venv\Scripts\python.exe flashlight_sandbox.py                 (uni profile, fullscreen)
  venv\Scripts\python.exe flashlight_sandbox.py --windowed      (desk testing)
  venv\Scripts\python.exe flashlight_sandbox.py --profile home
"""

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse
import json
import threading
import time

import cv2
import pygame

from game import config as cfg
from game import calibration as cal_mod
from game import capture as cap_mod
from game import detector as det_mod
from game.detector import DetectorStats

# Same profiles as game/main.py (kept in sync by hand).
PROFILES = {
    "home": {"YOLO_DEVICE": "cuda:0", "YOLO_IMGSZ": 640, "YOLO_CONF": 0.25,
             "DROIDCAM_DEVICE_INDEX": 0},
    "uni":  {"YOLO_DEVICE": "cpu",    "YOLO_IMGSZ": 512, "YOLO_CONF": 0.20,
             "DROIDCAM_DEVICE_INDEX": 2},
}

TUNING_FILE = "flashlight_tuning.json"


class SharedState:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.laser_state = None
        self.laser_lock = threading.Lock()
        self.detector_stats = DetectorStats()
        self.preview_frame = None
        self.preview_lock = threading.Lock()


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
    cap.release()
    raise RuntimeError(f"Cannot open camera {index}. Check IriunCam / index.")


def build_spotlight(radius: int, falloff: float) -> pygame.Surface:
    radius = max(8, int(radius))
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        a = int(255 * (1 - r / radius) ** falloff)
        pygame.draw.circle(surf, (255, 255, 200, a), (radius, radius), r)
    return surf


def build_grid() -> pygame.Surface:
    surf = pygame.Surface((cfg.GAME_W, cfg.GAME_H))
    surf.fill((10, 12, 24))
    step = 96
    for x in range(0, cfg.GAME_W + 1, step):
        pygame.draw.line(surf, (40, 90, 130), (x, 0), (x, cfg.GAME_H), 1)
    for y in range(0, cfg.GAME_H + 1, step):
        pygame.draw.line(surf, (40, 90, 130), (0, y), (cfg.GAME_W, y), 1)
    # brighter centre + quadrant target rings
    targets = [(cfg.GAME_W // 2, cfg.GAME_H // 2),
               (cfg.GAME_W // 4, cfg.GAME_H // 4),
               (3 * cfg.GAME_W // 4, cfg.GAME_H // 4),
               (cfg.GAME_W // 4, 3 * cfg.GAME_H // 4),
               (3 * cfg.GAME_W // 4, 3 * cfg.GAME_H // 4)]
    for (tx, ty) in targets:
        pygame.draw.circle(surf, (90, 200, 255), (tx, ty), 26, 2)
        pygame.draw.line(surf, (90, 200, 255), (tx - 16, ty), (tx + 16, ty), 1)
        pygame.draw.line(surf, (90, 200, 255), (tx, ty - 16), (tx, ty + 16), 1)
    return surf, targets


def draw_overlay(screen, x, y, spot_surf, radius, dark_alpha, hole_alpha):
    if dark_alpha <= 0:
        return
    overlay = pygame.Surface((cfg.GAME_W, cfg.GAME_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, int(dark_alpha)))
    if hole_alpha > 5:
        scaled = spot_surf.copy()
        scaled.fill((255, 255, 255, int(hole_alpha)), special_flags=pygame.BLEND_RGBA_MULT)
        overlay.blit(scaled, (int(x) - radius, int(y) - radius),
                     special_flags=pygame.BLEND_RGBA_SUB)
    screen.blit(overlay, (0, 0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="uni", choices=list(PROFILES))
    p.add_argument("--device", type=int, default=None, help="camera index override")
    p.add_argument("--windowed", action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    args = p.parse_args()

    for k, v in PROFILES[args.profile].items():
        setattr(cfg, k, v)
    if args.device is not None:
        cfg.DROIDCAM_DEVICE_INDEX = args.device
    print(f"[sandbox] profile={args.profile} cam={cfg.DROIDCAM_DEVICE_INDEX} "
          f"imgsz={cfg.YOLO_IMGSZ} device={cfg.YOLO_DEVICE}")

    # ── camera + calibration ────────────────────────────────────────────────
    cap = open_camera(cfg.DROIDCAM_DEVICE_INDEX)
    try:
        calibration = cal_mod.Calibration.load_or_run(cap, force=args.recalibrate)
    except RuntimeError as e:
        cap.release()
        print(f"[sandbox] calibration failed: {e}")
        return
    cap.release()
    time.sleep(0.3)

    shared = SharedState()
    stop_event = threading.Event()
    capture_thread = cap_mod.CaptureThread(shared, stop_event,
                                           device_index=cfg.DROIDCAM_DEVICE_INDEX)
    detector_thread = det_mod.DetectorThread(shared, stop_event, calibration, debug=False)
    capture_thread.start()
    detector_thread.start()

    # ── pygame ───────────────────────────────────────────────────────────────
    pygame.init()
    flags = 0 if args.windowed else pygame.FULLSCREEN
    size = (1280, 720) if args.windowed else (cfg.GAME_W, cfg.GAME_H)
    screen = pygame.display.set_mode(size, flags)
    pygame.display.set_caption("Flashlight sandbox")
    font = pygame.font.SysFont("monospace", 20)
    font_s = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    grid_surf, targets = build_grid()

    # tunable parameters
    params = [
        {"name": "radius",     "val": float(cfg.SPOTLIGHT_RADIUS), "min": 40,  "max": 600, "step": 5,    "fine": 1},
        {"name": "offset_x",   "val": 0.0,                          "min": -400,"max": 400, "step": 10,   "fine": 2},
        {"name": "offset_y",   "val": 0.0,                          "min": -400,"max": 400, "step": 10,   "fine": 2},
        {"name": "smoothing",  "val": 0.0,                          "min": 0.0, "max": 0.95,"step": 0.05, "fine": 0.01},
        {"name": "fade_secs",  "val": float(cfg.SPOTLIGHT_FADE_SECS),"min": 0.0,"max": 2.0, "step": 0.1,  "fine": 0.02},
        {"name": "stale_secs", "val": float(cfg.LASER_STALE_SECS),  "min": 0.05,"max": 1.0, "step": 0.05, "fine": 0.01},
        {"name": "falloff",    "val": 0.6,                          "min": 0.2, "max": 2.0, "step": 0.1,  "fine": 0.05},
        {"name": "dark_alpha", "val": 245.0,                        "min": 0,   "max": 255, "step": 5,    "fine": 1},
    ]
    sel = 0

    def pv(name):
        return next(p for p in params if p["name"] == name)["val"]

    # slider screen geometry (left column)
    panel_x, panel_y = 24, 90
    row_h = 40
    bar_x = panel_x + 150
    bar_w = 280
    slider_rects = []  # (index, pygame.Rect)
    for i in range(len(params)):
        slider_rects.append((i, pygame.Rect(bar_x, panel_y + i * row_h + 6, bar_w, 18)))

    spot_surf = build_spotlight(int(pv("radius")), pv("falloff"))
    last_radius, last_falloff = pv("radius"), pv("falloff")

    smoothed = [cfg.GAME_W / 2, cfg.GAME_H / 2]
    spot_alpha = 0.0
    show_grid = True
    dragging = None

    def set_from_mouse(idx, mx):
        pr = params[idx]
        rect = slider_rects[idx][1]
        frac = max(0.0, min(1.0, (mx - rect.x) / rect.w))
        pr["val"] = pr["min"] + frac * (pr["max"] - pr["min"])

    def save_values():
        data = {p["name"]: round(p["val"], 3) for p in params}
        with open(TUNING_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[sandbox] saved {TUNING_FILE}: {data}")
        print("[sandbox] suggested config.py / game_loop.py values:")
        print(f"    SPOTLIGHT_RADIUS    = {int(data['radius'])}")
        print(f"    SPOTLIGHT_FADE_SECS = {data['fade_secs']}")
        print(f"    LASER_STALE_SECS    = {data['stale_secs']}")
        print(f"    # spotlight offset correction (new): "
              f"({int(data['offset_x'])}, {int(data['offset_y'])})")
        print(f"    # smoothing={data['smoothing']}  falloff={data['falloff']}  "
              f"dark_alpha={int(data['dark_alpha'])}")

    running = True
    while running:
        dt = clock.tick(cfg.TARGET_FPS) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_DOWN:
                    sel = (sel + 1) % len(params)
                elif e.key == pygame.K_UP:
                    sel = (sel - 1) % len(params)
                elif e.key == pygame.K_LEFT:
                    params[sel]["val"] = max(params[sel]["min"], params[sel]["val"] - params[sel]["fine"])
                elif e.key == pygame.K_RIGHT:
                    params[sel]["val"] = min(params[sel]["max"], params[sel]["val"] + params[sel]["fine"])
                elif e.key == pygame.K_PAGEDOWN:
                    params[sel]["val"] = max(params[sel]["min"], params[sel]["val"] - params[sel]["step"])
                elif e.key == pygame.K_PAGEUP:
                    params[sel]["val"] = min(params[sel]["max"], params[sel]["val"] + params[sel]["step"])
                elif e.key == pygame.K_c:
                    for pr in params:
                        if pr["name"] in ("offset_x", "offset_y"):
                            pr["val"] = 0.0
                elif e.key == pygame.K_g:
                    show_grid = not show_grid
                elif e.key == pygame.K_s:
                    save_values()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for idx, rect in slider_rects:
                    if rect.collidepoint(e.pos):
                        dragging = idx
                        sel = idx
                        set_from_mouse(idx, e.pos[0])
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = None
            elif e.type == pygame.MOUSEMOTION and dragging is not None:
                set_from_mouse(dragging, e.pos[0])

        # rebuild spotlight if shape changed
        if pv("radius") != last_radius or pv("falloff") != last_falloff:
            spot_surf = build_spotlight(int(pv("radius")), pv("falloff"))
            last_radius, last_falloff = pv("radius"), pv("falloff")
        radius = int(pv("radius"))

        # ── laser ─────────────────────────────────────────────────────────────
        with shared.laser_lock:
            ls = shared.laser_state
        fresh = (ls is not None) and ((time.time() - ls.timestamp) < pv("stale_secs"))
        raw = ls.pos if ls is not None else None

        fade_rate = 255.0 / max(pv("fade_secs"), 0.01)
        if fresh:
            target = (raw[0] + pv("offset_x"), raw[1] + pv("offset_y"))
            sm = pv("smoothing")
            a = 1.0 if sm <= 0 else max(0.02, 1.0 - sm)
            smoothed[0] += (target[0] - smoothed[0]) * a
            smoothed[1] += (target[1] - smoothed[1]) * a
            spot_alpha = min(255.0, spot_alpha + fade_rate * dt * 4)
        else:
            spot_alpha = max(0.0, spot_alpha - fade_rate * dt)

        # ── render ──────────────────────────────────────────────────────────────
        if show_grid:
            screen.blit(pygame.transform.scale(grid_surf, screen.get_size()), (0, 0))
        else:
            screen.fill((6, 6, 14))

        sx = smoothed[0] * screen.get_width() / cfg.GAME_W
        sy = smoothed[1] * screen.get_height() / cfg.GAME_H
        draw_overlay(screen, sx, sy, spot_surf, radius,
                     dark_alpha=pv("dark_alpha"), hole_alpha=int(spot_alpha))

        # crosshairs (on top, always visible)
        scl_x = screen.get_width() / cfg.GAME_W
        scl_y = screen.get_height() / cfg.GAME_H
        if raw is not None:
            rx, ry = int(raw[0] * scl_x), int(raw[1] * scl_y)
            pygame.draw.line(screen, (255, 60, 60), (rx - 14, ry), (rx + 14, ry), 2)
            pygame.draw.line(screen, (255, 60, 60), (rx, ry - 14), (rx, ry + 14), 2)
        gx, gy = int(sx), int(sy)
        pygame.draw.line(screen, (60, 255, 60), (gx - 18, gy), (gx + 18, gy), 2)
        pygame.draw.line(screen, (60, 255, 60), (gx, gy - 18), (gx, gy + 18), 2)

        # control panel
        title = font.render("FLASHLIGHT SANDBOX  (red +=raw, green +=corrected)", True, (230, 230, 120))
        screen.blit(title, (panel_x, 50))
        for i, pr in enumerate(params):
            y = panel_y + i * row_h
            is_sel = (i == sel)
            color = (120, 255, 120) if is_sel else (200, 200, 200)
            label = font.render(f"{pr['name']}", True, color)
            screen.blit(label, (panel_x, y))
            rect = slider_rects[i][1]
            pygame.draw.rect(screen, (40, 40, 60), rect)
            frac = (pr["val"] - pr["min"]) / (pr["max"] - pr["min"])
            pygame.draw.rect(screen, color, (rect.x, rect.y, int(rect.w * frac), rect.h))
            valtxt = font_s.render(f"{pr['val']:.2f}", True, (230, 230, 230))
            screen.blit(valtxt, (rect.right + 10, y))

        # status
        stats = shared.detector_stats
        st = "DETECTING" if fresh else ("STALE" if ls is not None else "NO LASER YET")
        info = [
            f"detect={stats.detect_fps:4.1f}fps infer={stats.infer_fps:4.1f}fps  {st}",
            (f"raw=({raw[0]:.0f},{raw[1]:.0f})  corrected=({smoothed[0]:.0f},{smoothed[1]:.0f})"
             if raw is not None else "raw=(--)"),
            "UP/DOWN pick  LEFT/RIGHT fine  PgUp/PgDn coarse  drag=set  "
            "C=zero offset  G=grid  S=save  Q=quit  (restart w/ --recalibrate to re-cal)",
        ]
        for j, line in enumerate(info):
            t = font_s.render(line, True, (200, 255, 200))
            screen.blit(t, (panel_x, screen.get_height() - 70 + j * 20))

        pygame.display.flip()

    stop_event.set()
    capture_thread.join(timeout=3)
    detector_thread.join(timeout=3)
    pygame.quit()
    print("[sandbox] done")


if __name__ == "__main__":
    main()
