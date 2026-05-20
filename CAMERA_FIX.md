# DroidCam Green-Screen Fix

Copy-paste reference for fixing the DroidCam green-screen issue. Pull
latest first:

```
git pull
```

## CONFIRMED DIAGNOSIS (2026-05-20, i7 uni laptop)

`cam_diag.py` proved the green/frozen feed is NOT a colour bug:

- index 1 returns a constant `mean=45.33` across 60 frames (spread 0.00)
  = a FROZEN buffer, no flowing video.
- `fourcc=''` (empty) = OpenCV never negotiated a real video stream.
- DSHOW throws "can't capture by index" = DroidCam's DirectShow filter
  is not reachable by OpenCV's index scan.

Conclusions:
- Index 1 is most likely the laptop's **IR / Windows Hello camera**
  (static dark frame when idle), NOT DroidCam.
- **DroidCam is not enumerable by OpenCV by index on this laptop.**

=> The env-var / fourcc / backend tricks below CANNOT help, because
there is no stream to decode. The fix is the **OBS Virtual Camera
bridge** (Step OBS, below): OBS grabs DroidCam by name and republishes
a normal camera OpenCV can read.

---

## THE OBS VIRTUAL CAMERA BRIDGE (the actual solution)

1. Install OBS Studio if not present (free, ~120 MB, obsproject.com).
2. Open OBS.
3. In the **Sources** box (bottom), click **+** -> **Video Capture
   Device** -> name it "DroidCam" -> OK.
4. In Properties, open the **Device** dropdown and pick the entry whose
   name contains **DroidCam** (e.g. "DroidCam Source 1", "DroidCam
   Source 2", or "DroidCam Source 3"). Try each until the OBS preview
   shows the live phone feed.
5. Click **OK**. The OBS canvas must now show the live projector image.
   (If every DroidCam entry is black in OBS too, the phone-PC link is
   the problem — go to the Wi-Fi fallback at the bottom.)
6. In the **Controls** box (bottom-right) click **Start Virtual
   Camera**. The label flips to "Stop Virtual Camera".
7. **Leave OBS open the entire session.** Closing it kills the camera.
8. Back in the terminal:
   ```
   venv\Scripts\python.exe find_cam.py
   ```
   A NEW working camera appears (often the highest index) labelled OBS
   Virtual Camera, with a CHANGING mean (not frozen). Preview it:
   ```
   venv\Scripts\python.exe find_cam.py --preview 2 --backend AUTO
   venv\Scripts\python.exe find_cam.py --preview 3 --backend AUTO
   ```
9. When a preview shows the projector image, use that index everywhere:
   ```
   venv\Scripts\python.exe live_bg_test.py --device 2 --backend AUTO --hold
   venv\Scripts\python.exe -m game.main --profile uni --device 2 --recalibrate --debug
   ```
   No `--fourcc` needed — OBS Virtual Camera is well-behaved.

---

## (Older tricks — only relevant if a stream IS flowing) The env-var fix

The green/garbled feed is caused by OpenCV's Media Foundation (MSMF)
backend using **hardware transforms** that DroidCam's virtual camera
driver can't feed correctly. The fix is the environment variable
`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`.

**This is now baked into `find_cam.py`, `live_bg_test.py`, and
`game/main.py`** — they set it automatically before importing OpenCV.
So after `git pull`, just re-run find_cam and the MSMF feed should show
the real image instead of green:

```
venv\Scripts\python.exe find_cam.py
venv\Scripts\python.exe find_cam.py --preview 1 --backend MSMF
venv\Scripts\python.exe find_cam.py --preview 2 --backend MSMF
venv\Scripts\python.exe find_cam.py --preview 1 --backend AUTO
```

If the previously-green DroidCam index now shows the projector image,
you're done — use that index/backend:

```
venv\Scripts\python.exe live_bg_test.py --device 1 --backend MSMF --hold
```

If you ever need to set the variable manually in a fresh PowerShell
(e.g. for a script that doesn't set it itself):

```
$env:OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS = "0"
```

If that fixes it, skip the rest of this file. The MJPG / fourcc tricks
below are only fallbacks for when the env-var fix doesn't take.

---

## Context (older notes)

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
