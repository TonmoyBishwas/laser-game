"""
Generate a GREEN-LASER-SAFE diverse background pack into `bg_pack/` (1920x1080 PNG).

This complements make_bg_images.py (30 plain solids, some of them green — those
were for the OLD red laser). Everything here avoids bright/saturated green so the
green dot is always the only green thing on screen.

Categories (filename prefix):
  s_   non-green solids            (clean contrast test on every hue)
  g_   linear / radial gradients   (dot crosses brightness + hue ramps)
  t_   busy textures               (checker, stripes, noise, mosaic, bokeh)
  b_   bright / high-key scenes    (HARD case: low contrast, white bloom blends)
  d_   dark / low-key scenes       (easy case, matches the real maze)
  n_   negatives / distractors     (white glints, warm spots, NO green — teaches
                                    "a bright blob alone is not a laser")
  art_ real game art               (thief background tiled to 1080p)

  venv\\Scripts\\python.exe make_bg_pack.py
Then project bg_pack\\ as a looping slideshow (3-4 s per image) and film the laser.
"""

from pathlib import Path

import cv2
import numpy as np

W, H = 1920, 1080
OUT = Path("bg_pack")
rng = np.random.default_rng(7)


def rgb(r, g, b):
    return np.array([b, g, r], dtype=np.uint8)  # cv2 is BGR


def solid(c):
    img = np.empty((H, W, 3), np.uint8)
    img[:] = c
    return img


def lin_grad(c0, c1, horizontal=True):
    t = np.linspace(0, 1, W if horizontal else H, dtype=np.float32)
    ramp = (1 - t)[:, None] * c0.astype(np.float32) + t[:, None] * c1.astype(np.float32)
    img = np.broadcast_to(ramp[None, :, :] if horizontal else ramp[:, None, :], (H, W, 3))
    return img.astype(np.uint8).copy()


def rad_grad(c_center, c_edge):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
    t = np.clip(r / r.max(), 0, 1)[..., None]
    return ((1 - t) * c_center + t * c_edge).astype(np.uint8)


def checker(c0, c1, size):
    yy, xx = np.mgrid[0:H, 0:W]
    m = ((yy // size + xx // size) % 2 == 0)[..., None]
    return np.where(m, c0, c1).astype(np.uint8)


def stripes(c0, c1, size, diag=False):
    yy, xx = np.mgrid[0:H, 0:W]
    v = (xx + yy) if diag else xx
    m = ((v // size) % 2 == 0)[..., None]
    return np.where(m, c0, c1).astype(np.uint8)


def noise(lo, hi, blur=0, tint=None):
    img = rng.integers(lo, hi, (H, W, 3), dtype=np.uint8)
    if tint is not None:
        img = (img.astype(np.float32) * tint / 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def mosaic(palette, cell):
    tiles = rng.integers(0, len(palette), (H // cell + 1, W // cell + 1))
    img = np.array(palette, np.uint8)[tiles]
    return cv2.resize(img, (img.shape[1] * cell, img.shape[0] * cell),
                      interpolation=cv2.INTER_NEAREST)[:H, :W]


def bokeh(bg, colors, n, rmin, rmax):
    img = solid(bg)
    for _ in range(n):
        c = colors[rng.integers(len(colors))]
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        r = int(rng.integers(rmin, rmax))
        layer = img.copy()
        cv2.circle(layer, (x, y), r, tuple(int(v) for v in c), -1, cv2.LINE_AA)
        img = cv2.addWeighted(img, 0.45, layer, 0.55, 0)
    return cv2.GaussianBlur(img, (0, 0), 6)


def glints(bg, n, rmin, rmax, colors):
    """Small hard-edged bright spots (white/warm) — laser look-alikes with no green."""
    img = solid(bg)
    for _ in range(n):
        x, y = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        r = int(rng.integers(rmin, rmax))
        c = colors[rng.integers(len(colors))]
        cv2.circle(img, (x, y), r, tuple(int(v) for v in c), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def is_greenish(c):
    b, g, r = int(c[0]), int(c[1]), int(c[2])
    return g > 120 and g > r + 40 and g > b + 40


def main():
    OUT.mkdir(exist_ok=True)
    K = rgb(0, 0, 0); WH = rgb(255, 255, 255)
    RED = rgb(220, 30, 30); ORG = rgb(255, 140, 0); YEL = rgb(255, 220, 0)
    BLU = rgb(30, 60, 220); CYA = rgb(0, 190, 230); MAG = rgb(210, 30, 200)
    PUR = rgb(90, 20, 160); PNK = rgb(255, 120, 180); BRN = rgb(120, 70, 30)
    NAV = rgb(8, 12, 48); GRY = rgb(128, 128, 128); LGR = rgb(215, 215, 215)
    CRM = rgb(250, 240, 210); SKY = rgb(150, 200, 255)

    imgs = {}
    # ── solids ──
    for name, c in [("red", RED), ("orange", ORG), ("yellow", YEL), ("blue", BLU),
                    ("cyan", CYA), ("magenta", MAG), ("purple", PUR), ("pink", PNK),
                    ("brown", BRN), ("white", WH), ("cream", CRM), ("gray50", GRY),
                    ("lightgray", LGR), ("navy", NAV), ("black", K)]:
        imgs[f"s_{name}"] = solid(c)
    # ── gradients ──
    imgs["g_black_white_h"] = lin_grad(K, WH)
    imgs["g_black_white_v"] = lin_grad(K, WH, False)
    imgs["g_blue_orange"] = lin_grad(BLU, ORG)
    imgs["g_purple_yellow"] = lin_grad(PUR, YEL)
    imgs["g_red_cyan_v"] = lin_grad(RED, CYA, False)
    imgs["g_navy_pink"] = lin_grad(NAV, PNK)
    imgs["g_radial_white_navy"] = rad_grad(WH, NAV)
    imgs["g_radial_yellow_red"] = rad_grad(YEL, RED)
    imgs["g_radial_sky_purple"] = rad_grad(SKY, PUR)
    # ── textures ──
    imgs["t_checker_bw_120"] = checker(K, WH, 120)
    imgs["t_checker_bw_30"] = checker(K, WH, 30)
    imgs["t_checker_red_blue"] = checker(RED, BLU, 80)
    imgs["t_stripes_bw"] = stripes(K, WH, 40)
    imgs["t_stripes_diag_or_pu"] = stripes(ORG, PUR, 60, True)
    imgs["t_noise_fine"] = noise(0, 255)
    imgs["t_noise_soft"] = noise(40, 220, blur=8)
    imgs["t_noise_warm"] = noise(60, 255, blur=3, tint=np.array([120, 170, 255]))
    imgs["t_noise_cool"] = noise(60, 255, blur=3, tint=np.array([255, 170, 120]))
    imgs["t_mosaic_warm"] = mosaic([RED, ORG, YEL, BRN, MAG, WH], 90)
    imgs["t_mosaic_cool"] = mosaic([BLU, CYA, PUR, NAV, WH, GRY], 60)
    imgs["t_mosaic_pastel"] = mosaic([CRM, PNK, SKY, LGR, YEL], 45)
    imgs["t_bokeh_neon"] = bokeh(NAV, [MAG, CYA, ORG, PNK, YEL], 60, 40, 160)
    imgs["t_bokeh_warm"] = bokeh(BRN, [ORG, YEL, RED, CRM], 50, 50, 200)
    # ── bright / high-key ──
    imgs["b_white"] = solid(WH)
    imgs["b_cream_soft"] = noise(220, 256, blur=10)
    imgs["b_sky_radial"] = rad_grad(WH, SKY)
    imgs["b_yellow_white"] = lin_grad(YEL, WH)
    imgs["b_cyan_white"] = rad_grad(CYA, WH)
    imgs["b_pink_bokeh"] = bokeh(CRM, [PNK, YEL, WH, SKY], 45, 80, 250)
    imgs["b_snow_noise"] = noise(200, 256, blur=2)
    # ── dark / low-key ──
    imgs["d_black"] = solid(K)
    imgs["d_navy_stars"] = glints(NAV, 250, 1, 3, [WH, SKY, CRM])
    imgs["d_dungeon_noise"] = noise(0, 70, blur=4, tint=np.array([150, 170, 255]))
    imgs["d_purple_radial"] = rad_grad(PUR, K)
    imgs["d_ember_bokeh"] = bokeh(K, [RED, ORG, BRN], 35, 40, 150)
    imgs["d_gray_checker"] = checker(rgb(25, 25, 25), rgb(55, 55, 55), 100)
    # ── negatives / distractors (no green anywhere) ──
    imgs["n_white_glints_dark"] = glints(NAV, 25, 6, 22, [WH, CRM])
    imgs["n_white_glints_mid"] = glints(GRY, 25, 6, 22, [WH])
    imgs["n_warm_spots_black"] = glints(K, 25, 6, 22, [YEL, ORG, WH])
    imgs["n_cyan_spots_black"] = glints(K, 25, 6, 22, [CYA, SKY, WH])
    imgs["n_white_dots_busy"] = glints(rgb(90, 40, 110), 40, 4, 16, [WH, PNK, YEL])

    # ── real game art ──
    art = Path("arcade_game/games/thief/assets/background.png")
    if art.exists():
        a = cv2.imread(str(art))
        if a is not None:
            imgs["art_thief_bg_scaled"] = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
            reps = (H // a.shape[0] + 1, W // a.shape[1] + 1, 1)
            imgs["art_thief_bg_tiled"] = np.tile(a, reps)[:H, :W]

    n = 0
    for name, img in imgs.items():
        mean = img.reshape(-1, 3).mean(0)
        if is_greenish(mean):
            print(f"SKIP (greenish): {name}")
            continue
        cv2.imwrite(str(OUT / f"{name}.png"), img)
        n += 1
    print(f"wrote {n} images to {OUT.resolve()}")


if __name__ == "__main__":
    main()
