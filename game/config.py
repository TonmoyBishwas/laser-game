import os

# ── Display ────────────────────────────────────────────────────────────────────
GAME_W, GAME_H = 1920, 1080
TARGET_FPS = 60

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    r"runs\detect\laser_detector_run\weights\best.pt",
)
YOLO_CONF   = 0.25
YOLO_IMGSZ  = 640   # match training resolution for best recall; lower if FPS suffers
YOLO_DEVICE = "cuda:0"

# ── Laser color gate ───────────────────────────────────────────────────────────
# The detector model fires on any bright blob, so bright projected art (white HUD
# text, the red edge glow) could be mistaken for the laser. The laser is GREEN, so
# reject any detection box that has essentially no green pixels. These thresholds
# are deliberately LENIENT: the gate's job now is only to reject obviously-non-green
# art, NOT to be a tight green filter. The treasure (the original reason for this
# gate) is now hidden in the dark by the game itself, so the gate no longer has to
# be strict — and a too-strict gate is worse than no gate, because a real laser
# overexposes toward a white/yellow-green core and a strict gate REJECTS it on
# borderline frames, making the spotlight flicker on/off (uncatchable). Disable
# entirely with --no-color-gate if a washed-out laser ever still drops out.
LASER_COLOR_GATE      = True
# Which laser is in use: "green" (deployed YOLO26-nano) or "red" (the older
# YOLO26-S in runs/detect/laser_detector_run_red). Switch at launch with
# --laser red|green; that swaps MODEL_PATH AND the colour gate together.
LASER_COLOR           = "green"
# Red gate: OpenCV hue wraps at 180, so a red dot reads ~0-10 OR ~170-179. The
# blown-out core is white; the halo is what these bands catch. (Same S/V/pixel
# thresholds as green.)
RED_GATE_H_BANDS      = ((0, 10), (170, 179))
GREEN_GATE_H_LO       = 35    # OpenCV hue (0-179); the 532nm laser films ~60 but an
                              # overexposed dot can read down into the 35-45 range, so
                              # keep the low edge at 35 (anything stricter clipped the
                              # real laser and caused on/off flicker). Gold ~25 and red
                              # ~0/179 still fall outside; and the treasure is hidden in
                              # the dark now anyway, so it is no longer a false target.
GREEN_GATE_H_HI       = 95    # a touch wider on the cyan side for cool/blown-out green
GREEN_GATE_S_MIN      = 25    # min saturation (lowered: a blown-out green dot is paler)
GREEN_GATE_V_MIN      = 40    # min brightness
GREEN_GATE_MIN_PIXELS = 3     # need at least this many green px in the box
GREEN_GATE_MIN_FRACTION = 0.02  # ...and only this small a share of the box (the green
                                # halo around an overexposed white core can be thin)
# Cap detector inference rate (fps). 0 = unlimited. On CPU-only machines a fast
# backend (onnx/openvino) can run inference back-to-back and starve the render
# thread of CPU; capping leaves gaps so the spotlight redraws smoothly.
MAX_DETECT_FPS = 0

# ── Camera ─────────────────────────────────────────────────────────────────────
DROIDCAM_DEVICE_INDEX = 0

# ── Calibration ────────────────────────────────────────────────────────────────
CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "calibration.json"
)

# ── Spotlight ──────────────────────────────────────────────────────────────────
SPOTLIGHT_RADIUS    = 160   # game pixels — generous so player can find the thief
SPOTLIGHT_FADE_SECS = 0.3
LASER_STALE_SECS    = 0.55   # hold the last laser spot this long before the spotlight
                             # fades. MUST be > 2 detector intervals: with
                             # MAX_DETECT_FPS 12 one inference every ~83 ms, so the old
                             # 0.15 s let a SINGLE missed detection blank the pointer
                             # (the on/off flicker seen on the i7). 0.55 s (field-tested
                             # on the i7, 2026-09-29) bridges ~6 missed inferences.
                             # Higher = more bridging but more lag on fast laser moves.

# ── Maze ───────────────────────────────────────────────────────────────────────
GRID_COLS, GRID_ROWS = 15, 9
CELL_W = GAME_W // GRID_COLS     # 128
CELL_H = GAME_H // GRID_ROWS     # 120
CELL_TRANSITION_SECS = 1.20      # ~0.83 cells per second; full path ~24 s

# ── Thief sprite ───────────────────────────────────────────────────────────────
THIEF_W, THIEF_H = 60, 80

# ── Treasure ───────────────────────────────────────────────────────────────────
TREASURE_W, TREASURE_H = 90, 70

# ── Catch mechanic ─────────────────────────────────────────────────────────────
CATCH_HOLD_SECS = 1.0    # how long the laser must stay on the thief
CATCH_RADIUS    = 100    # game pixels — slightly smaller than SPOTLIGHT_RADIUS
CATCH_DECAY_RATIO = 0.5  # catch_progress decays at this fraction of fill speed

# ── Edge glow (thief direction hint) ───────────────────────────────────────────
EDGE_GLOW_DEPTH = 60     # px — thickness of the glow band
EDGE_GLOW_BASE  = 0.45
EDGE_GLOW_PULSE = 0.55

# ── RL policy file ─────────────────────────────────────────────────────────────
POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "policies", "thief_qtable.npz"
)

# ── Timing ─────────────────────────────────────────────────────────────────────
ROUND_TIMEOUT_SECS = 90   # if neither side wins by then, round resets (debug)
