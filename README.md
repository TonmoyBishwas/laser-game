# Laser Guard

Interactive projector game where a player uses a laser pointer as a flashlight
to find and catch an RL-trained thief moving through a maze toward a treasure.

## Pipeline

```
Phone camera ──USB──> DroidCam virtual webcam ──> PC
                                                   ├── YOLOv8 fine-tuned for laser detection
                                                   ├── Homography: camera pixel → game coord
                                                   ├── Q-learned thief policy
                                                   └── Pygame at 60 FPS over the projector
```

## Quick start

### Home / dev PC (NVIDIA GPU)

```
setup_home.bat                                 # one-time
venv\Scripts\python.exe train_thief.py         # one-time: train RL policy
venv\Scripts\python.exe -m game.main --debug   # play
```

### Uni / friend's laptop (CPU only)

```
setup_uni.bat                                                       # one-time
venv\Scripts\python.exe -m game.main --profile uni --recalibrate    # first run
venv\Scripts\python.exe -m game.main --profile uni                  # subsequent runs
```

See **[README_UNI.md](README_UNI.md)** for the full uni / demo-day checklist.

## Tools

| Script | What it does |
|---|---|
| `game/main.py` | Run the game |
| `train_thief.py` | Train the Q-learning policy (~5 s on GPU) |
| `tune_detect.py` | Live or replay slider tuning of YOLO thresholds |
| `record_color_clips.py` | Records 10 solid-colour background clips for detection analysis |
| `find_cam.py` | Enumerates camera devices to find DroidCam |

## CLI flags (`game.main`)

- `--profile {home,uni}` — locks device + imgsz + conf appropriately
- `--debug` — YOLO preview window, HUD stats, Q-value bars, thief silhouette
- `--recalibrate` — forces the 4-corner homography setup
- `--device N` — pick camera device index (default 0)
- `--fullscreen` — borderless full-screen on the projector

## Architecture

```
game/
├── config.py          all constants, mutable by --profile
├── calibration.py     4-corner click → cv2.findHomography → JSON
├── capture.py         DroidCam read thread (latest-frame only)
├── detector.py        YOLO inference thread → game-space laser coord
├── maze.py            grid + lookup helpers
├── maze_layouts.py    hand-designed level (edge-biased)
├── thief.py           cell-based motion driven by the policy
├── rl_env.py          gym-style training environment
├── rl_policy.py       tabular Q-table — save/load/act/update
├── game_loop.py       pygame render: dark overlay, flashlight, edge glow
└── main.py            entry point + thread orchestration
```

## RL approach (in 30 seconds)

Tabular Q-learning. State = `(thief_row, thief_col, laser_row, laser_col)`. 4
actions (N/E/S/W). Reward shaping: +1 reaching treasure, -1 if caught, -0.01
per step, -0.05 per wall hit. 20 000 training episodes, ε-greedy from 1.0 to
0.05 over 80 % of training. Trained against four laser modes (idle, patrol,
tracking-with-lag, random). Final win rate vs. the scripted laser: ~95 %.
