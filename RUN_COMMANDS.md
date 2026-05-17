# Laser Guard — Run Commands

Copy-paste reference for running the BG tests and the main game on the
friend's i3 uni laptop. No edits needed unless you want to change flags.

All commands are run from the project root (`H:\...\laser-game\` or
wherever you cloned). The venv path is `venv\Scripts\python.exe` on
Windows.

---

## 1. One-time per laptop

```
git pull
setup_uni.bat
```

`setup_uni.bat` installs CPU torch + everything else into `venv\`
(~200 MB, ~5 min on uni Wi-Fi).

---

## 2. Generate the solid-colour PNGs (optional)

```
venv\Scripts\python.exe make_bg_images.py
```

Writes `backgrounds\01_black.png` … `30_dark_olive.png` (1920x1080). Open
any in Windows Photos and press **F11** for fullscreen if you want to
inspect a colour by eye later. Folder is gitignored.

---

## 3. Auto leaderboard test (no quiet phase by default)

Each colour gets a 15 s ACTIVE phase — just sweep the laser around the
projector the whole time (centre → corners → edges → fast + slow).
After all colours finish, a sorted leaderboard is printed and saved.

```
:: All 30 colours, ~7.5 min total
venv\Scripts\python.exe live_bg_test.py

:: Only the 20 DARKER additions (skip the first 10 originals), ~5 min
venv\Scripts\python.exe live_bg_test.py --start 10

:: Only the first 10 originals for comparison, ~2.5 min
venv\Scripts\python.exe live_bg_test.py --limit 10

:: Slice indices 11..20 only (darker greys + dark blues)
venv\Scripts\python.exe live_bg_test.py --start 10 --limit 10

:: Shorter per-colour timing (faster total run)
venv\Scripts\python.exe live_bg_test.py --start 10 --active 10

:: Add a 2 s false-positive measurement at the start of each colour
:: (laser pointed at the floor during the QUIET phase)
venv\Scripts\python.exe live_bg_test.py --quiet 2 --active 15
```

Keys during the run:

| key   | action |
|-------|---|
| SPACE | skip current colour |
| ESC   | abort the run (still saves rows completed so far) |

Output written to:

- `bg_test_results\latest.txt` — readable sorted leaderboard
- `bg_test_results\latest.json` — full numbers, all rows
- `bg_test_results\results_<timestamp>.{json,txt}` — timestamped copies

---

## 4. Hold mode — manual eyeballing with arrow keys

Fullscreens one colour and runs detection forever. Use this to confirm
the winner from step 3, or to compare 2-3 contenders side-by-side. The
projector shows the colour + a small text panel + a 30-cell history
strip + a big green pulsing indicator (top-right) that flashes on every
detection. No infinity mirror — only text/UI is drawn, never the camera
feed.

```
:: Start at colour 1 and arrow through the palette
venv\Scripts\python.exe live_bg_test.py --hold

:: Start at a specific colour
venv\Scripts\python.exe live_bg_test.py --hold 15_dark_blue
venv\Scripts\python.exe live_bg_test.py --hold 22_forest

:: Custom RGB (arrows then jump into the palette)
venv\Scripts\python.exe live_bg_test.py --hold 30,30,80
```

Keys:

| key             | action |
|-----------------|---|
| ← / →           | previous / next colour |
| PgUp / PgDn     | jump ±5 colours |
| Home / End      | first / last colour |
| R               | reset stats (same colour) |
| ESC             | exit (final stats printed to console) |

Reading the **history strip** (30 cells under the text panel):

- All green → solid detection (best)
- All red → not detecting at all
- Green/red checker → flaky
- Green, gap of red, green again → "lost and recovered" (worst for gameplay — spotlight will flicker)

The `last30=NN%` number in the live line is the rolling detection rate
over those 30 frames. Stable colours sit at 90-100 %.

---

## 5. Pick the winning colour and edit the game

1. Open `bg_test_results\latest.txt`, read the top row.
2. Open `game\game_loop.py` in any editor.
3. Find the line `BG_COLOR = (...)` (Ctrl+F for `BG_COLOR`).
4. Replace with the winner's RGB. Example for `15_dark_blue`:

   ```python
   BG_COLOR = (0, 0, 64)
   ```

5. Save.

---

## 6. Run the main game

```
:: First boot on this laptop (4-corner calibration required)
venv\Scripts\python.exe -m game.main --profile uni --recalibrate --debug

:: Every run after that (calibration cached in calibration.json)
venv\Scripts\python.exe -m game.main --profile uni --debug

:: Demo run, no debug HUD
venv\Scripts\python.exe -m game.main --profile uni

:: If find_cam.py shows DroidCam at a different index
venv\Scripts\python.exe -m game.main --profile uni --debug --device 1
```

Calibration click order: **TL → TR → BR → BL**, then **Enter**.

`--profile uni` forces YOLO to: device=cpu, imgsz=320, conf=0.20 — the
tuned-for-i3 settings.

---

## 7. Camera troubleshooting

```
:: Enumerate cameras to find DroidCam's index (usually 0)
venv\Scripts\python.exe find_cam.py
```

If DroidCam isn't appearing in `find_cam.py`:

1. Phone DroidCam app open.
2. Windows DroidCam app open, USB tab clicked.
3. Phone is plugged in with the USB cable.
4. You can see the projector image in DroidCam's PC preview window.

---

## Suggested tonight-at-uni flow

```
:: 1. (Optional) generate PNGs for later inspection
venv\Scripts\python.exe make_bg_images.py

:: 2. Auto-test only the darker additions (~5 min)
venv\Scripts\python.exe live_bg_test.py --start 10

:: 3. Confirm the winner manually (compare top 2-3 with arrows)
venv\Scripts\python.exe live_bg_test.py --hold 15_dark_blue

:: 4. Edit game/game_loop.py: BG_COLOR = (R, G, B)

:: 5. Calibrate + play
venv\Scripts\python.exe -m game.main --profile uni --recalibrate --debug
```

---

## Cheat sheet — flags reference

`live_bg_test.py`:

| flag              | default | meaning |
|-------------------|---------|---------|
| `--device N`      | 0       | camera index (DroidCam usually 0) |
| `--quiet S`       | 0       | seconds of false-positive measurement (0 = skip) |
| `--active S`      | 15      | seconds of laser-sweeping per colour |
| `--conf F`        | 0.20    | YOLO confidence threshold |
| `--imgsz N`       | 320     | YOLO input size (320 is CPU-friendly) |
| `--start N`       | 0       | skip the first N colours |
| `--limit N`       | 0       | only test the first N (after `--start`) |
| `--hold [VAL]`    | off     | hold mode (browse / name / RGB) |
| `--windowed`      | off     | windowed mode for desk testing |
| `--outdir DIR`    | `bg_test_results` | where to save results |

`game.main`:

| flag              | default | meaning |
|-------------------|---------|---------|
| `--profile {home,uni}` | none | locks device + imgsz + conf for the hardware |
| `--recalibrate`   | off     | force the 4-corner setup |
| `--debug`         | off     | YOLO preview + HUD stats + Q-value bars |
| `--device N`      | 0       | camera index |
| `--fullscreen`    | off     | borderless fullscreen on the projector |
