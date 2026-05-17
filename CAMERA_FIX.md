# DroidCam Green-Screen Fix

Copy-paste reference for fixing the DroidCam green-screen issue on the
friend's i3 laptop. Pull latest first:

```
git pull
```

## Context

- `index=0` = laptop's built-in webcam (mean ~125, bright)
- `index=1` = DroidCam virtual camera (currently green-screen bug)
- `index=2` = OBS Virtual Camera (gives "disconnected" error — ignore)

So index 1 is the one we need to fix. The green screen is OpenCV
decoding DroidCam's pixel format wrong. The fix is to force MJPG via
`--fourcc MJPG`. Try the previews in order until one shows the
projector image (not green).

---

## Step 1 — find a preview that ISN'T green

Run these one at a time. Press `q` in each preview window to close
before running the next:

```
venv\Scripts\python.exe find_cam.py --preview 1 --backend DSHOW --fourcc MJPG
venv\Scripts\python.exe find_cam.py --preview 1 --backend AUTO --fourcc MJPG
venv\Scripts\python.exe find_cam.py --preview 1 --backend MSMF --fourcc MJPG
```

If all three still show green, also force the resolution:

```
venv\Scripts\python.exe find_cam.py --preview 1 --backend DSHOW --fourcc MJPG --width 640 --height 480
venv\Scripts\python.exe find_cam.py --preview 1 --backend AUTO --fourcc MJPG --width 640 --height 480
venv\Scripts\python.exe find_cam.py --preview 1 --backend MSMF --fourcc MJPG --width 640 --height 480
```

If still green, try YUY2 instead of MJPG:

```
venv\Scripts\python.exe find_cam.py --preview 1 --backend DSHOW --fourcc YUY2
venv\Scripts\python.exe find_cam.py --preview 1 --backend AUTO --fourcc YUY2
```

Or NV12:

```
venv\Scripts\python.exe find_cam.py --preview 1 --backend DSHOW --fourcc NV12
```

One of these combinations will show the projector image. Note down the
exact `--backend` and `--fourcc` values that worked.

---

## Step 2 — use those exact flags in live_bg_test

Replace `BACKEND` and `FOURCC` below with whatever worked in Step 1.

### Hold mode (manual eyeballing)

```
venv\Scripts\python.exe live_bg_test.py --device 1 --backend DSHOW --fourcc MJPG --hold
```

Or with resolution if Step 1 needed it:

```
venv\Scripts\python.exe live_bg_test.py --device 1 --backend DSHOW --fourcc MJPG --cam-width 640 --cam-height 480 --hold
```

### Auto leaderboard (darker colours only)

```
venv\Scripts\python.exe live_bg_test.py --device 1 --backend DSHOW --fourcc MJPG --start 10
```

### Main game

`game.main` doesn't accept fourcc flags yet — if Step 1 required MJPG,
the game itself will still hit the green screen. Tell me what worked
in Step 1 and I'll wire the same flags into `game/capture.py`.

---

## Step 3 — if NOTHING works (still all green)

Two reliable fallbacks:

### Option A — DroidCam over Wi-Fi

1. Disconnect the USB cable on the phone DroidCam app.
2. Phone and laptop must be on the same Wi-Fi network.
3. In DroidCam PC client, switch from USB tab to Wi-Fi tab.
4. Enter the phone's IP from the phone DroidCam app.
5. Click Start.
6. The PC client should show the live feed.
7. Re-run `find_cam.py` — Wi-Fi DroidCam usually doesn't hit the green
   bug because it uses MJPEG over HTTP natively.

### Option B — OBS Virtual Camera bridge

The laptop already has OBS. Use it to convert DroidCam → OBS Virtual
Camera (which OpenCV handles correctly):

1. Open OBS.
2. Add a new source → "Video Capture Device".
3. Select "DroidCam Source 1" (or whatever name DroidCam shows up as).
4. In OBS, the preview should show the projector.
5. Click "Start Virtual Camera" in OBS (bottom-right).
6. Make sure no other app (Zoom, Teams) is grabbing OBS Virtual Camera.
7. Re-run `find_cam.py`. OBS Virtual Camera will appear at some index
   (often 2). Test it:

   ```
   venv\Scripts\python.exe find_cam.py --preview 2 --backend AUTO
   ```

8. If it shows the projector image, use that index:

   ```
   venv\Scripts\python.exe live_bg_test.py --device 2 --backend AUTO --hold
   ```

---

## Cheat sheet — all the flags

| flag             | what it does |
|------------------|--------------|
| `--device N`     | camera index (0 = webcam, 1 = DroidCam, 2 = OBS) |
| `--backend B`    | `AUTO`, `DSHOW`, or `MSMF` |
| `--fourcc F`     | force pixel format: `MJPG`, `YUY2`, `NV12` |
| `--width N`      | (find_cam) force capture width |
| `--height N`     | (find_cam) force capture height |
| `--cam-width N`  | (live_bg_test) force capture width |
| `--cam-height N` | (live_bg_test) force capture height |
| `--hold [VAL]`   | (live_bg_test) lock to one colour, arrow nav |
| `--start N`      | (live_bg_test) skip first N colours |
| `--limit N`      | (live_bg_test) only test first N colours |
| `--conf F`       | (live_bg_test) YOLO confidence threshold (try 0.05) |
| `--imgsz N`      | (live_bg_test) YOLO input size (try 640) |
