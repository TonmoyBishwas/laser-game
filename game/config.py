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
LASER_STALE_SECS    = 0.10

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
