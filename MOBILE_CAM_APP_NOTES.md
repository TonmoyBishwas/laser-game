# Mobile Camera App — Build Notes for a DroidCam Replacement

> **Read this first.** This file is a hand-off document. It was written by a
> Claude Code instance that spent a long debugging session trying (and
> ultimately failing) to make **DroidCam** work as a USB webcam for a PC
> computer-vision project, across three different Windows machines. The goal of
> this document is to give a *fresh* Claude Code instance everything it needs to
> build a small **Android** camera-streaming app that **does not repeat
> DroidCam's mistakes**.
>
> Target platform: **Android only.**
> Chosen transport: **MJPEG over HTTP, USB (via `adb forward`) primary +
> Wi-Fi fallback. NO Windows virtual-camera driver.**

---

## 1. Purpose & the pipeline this feeds

There is a PC game ("Laser Guard") that uses a phone as a camera:

```
Phone on a tripod  ──►  points at a projector screen
        │
        ▼  (this app's job: get these frames to the PC)
PC receives video frames
        │
        ├─ OpenCV reads frames                 (cv2.VideoCapture)
        ├─ YOLOv26 detects a red laser dot      (the player's "flashlight")
        ├─ homography maps camera px → game px  (4-corner calibration)
        └─ Pygame renders the game on the projector at 60 FPS
```

The **only** thing the mobile app must do well: **deliver a low-latency,
reasonable-resolution camera stream to a Windows PC in a format OpenCV can read
reliably.** Everything else (detection, game) already exists and works.

Where it plugs in on the PC today (`game/capture.py`):

```python
# current code reads from a webcam DEVICE INDEX:
cap = cv2.VideoCapture(self._device, cv2.CAP_ANY)   # self._device is an int
```

Key fact that drives the whole recommended design: **OpenCV's
`cv2.VideoCapture` can take a URL string instead of an int**, and it reads an
**MJPEG-over-HTTP** stream natively:

```python
cap = cv2.VideoCapture("http://localhost:8080/video")   # MJPEG stream — just works
```

So if the app serves MJPEG over HTTP, the PC-side change is a one-liner and we
**completely avoid the Windows virtual-camera driver layer** that broke
DroidCam.

---

## 2. What went wrong with DroidCam (factual log)

These are the issues we hit, roughly in the order we hit them, across a home PC
(RTX 3090) and two uni laptops (i3, i7 — both no GPU).

1. **DSHOW backend throws a hard error.** `cv2.VideoCapture(idx, cv2.CAP_DSHOW)`
   raised: `VIDEOIO(DSHOW): raised unknown C++ exception!` and
   `backend is generally available but can't be used to capture by index`.
   DSHOW was effectively unusable for DroidCam on every machine.

2. **MSMF backend opens the *wrong* device.** Microsoft Media Foundation would
   "open" something, but it was a different/black virtual device, not the live
   feed.

3. **`CAP_ANY` worked on the home PC — but only there, and "by itself".** On the
   RTX 3090 box, `cv2.VideoCapture(0, cv2.CAP_ANY)` eventually worked. The user
   reports it "got fixed on its own" — i.e. it was never a deliberate, portable
   fix. It did **not** transfer to either laptop.

4. **Picking the wrong camera.** On laptops, `index 0` was the **built-in
   webcam** (bright, mean ≈ 130), not DroidCam. A second index showed a dim,
   frozen frame (mean ≈ 45.3) that was almost certainly the laptop's **IR /
   Windows Hello camera**, not DroidCam at all. DroidCam was **not reliably
   enumerable by OpenCV index** on the laptops.

5. **The blocking issue: a solid GREEN / frozen feed.** Whatever index seemed
   to be DroidCam returned a constant green image.

6. **We proved it was frozen, not just mis-coloured.** A diagnostic
   (`cam_diag.py`) read 60 consecutive frames and measured the per-frame mean
   brightness:
   - mean was **exactly `45.33` on all 60 frames** (spread `0.00`),
   - the negotiated pixel format was **empty: `fourcc=''`**.
   - Conclusion: OpenCV never negotiated a real video stream; the device hands
     back a **static placeholder buffer**. A live feed's mean always varies.

7. **The decisive test: OBS *also* gets green.** We tried using OBS Studio as a
   bridge. Selecting DroidCam as an OBS **"Video Capture Device"** *also* showed
   green — **but the DroidCam client app's own window showed the live feed
   perfectly.** That isolates the fault precisely:
   - `phone → USB → DroidCam PC client` = **works** (client window is live),
   - `DroidCam PC client → "DroidCam Source" virtual webcam device` =
     **broken** (emits a frozen/green buffer to *every* consumer: OBS & OpenCV).

---

## 3. Everything we tried (and the result)

| Attempt | Result |
|---|---|
| `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` env var, set before `import cv2` | No effect on the frozen feed |
| Force `CAP_PROP_FOURCC` = `MJPG` | No effect |
| Force `CAP_PROP_FOURCC` = `YUY2` | No effect |
| Force `CAP_PROP_FOURCC` = `NV12` | No effect |
| Force resolution `640×480` (`CAP_PROP_FRAME_WIDTH/HEIGHT`) | No effect |
| Force `CAP_PROP_CONVERT_RGB = 1` | No effect (MSMF logged "Failed to select stream 0") |
| Backend `AUTO` / `DSHOW` / `MSMF` | DSHOW unusable; AUTO & MSMF frozen |
| Try multiple device indices (0–4) | Only webcam + a frozen device found |
| OBS → "Video Capture Device" → DroidCam Source | Also green |
| OBS → **Window Capture** of the DroidCam client window → OBS Virtual Camera | The recommended *stopgap* — works because the client window itself renders the feed; OpenCV then reads the well-behaved OBS Virtual Camera |
| DroidCam over Wi-Fi instead of USB | Suggested as a fallback (Wi-Fi uses MJPEG-over-HTTP natively, which is exactly why it tends to "just work") |

The Wi-Fi-works / virtual-cam-fails split is the biggest hint of all: **the
reliable path is an HTTP stream, the unreliable path is a virtual-camera
driver.** That directly informs the architecture below.

---

## 4. Root-cause conclusion

The fragile link is the **Windows virtual-camera driver**. It fails for a
combination of reasons that are hard to control from the PC side:

- **Pixel-format negotiation** between the virtual device and the consumer
  (DirectShow/MSMF) is brittle; when it fails you get green/garbage or a frozen
  buffer **with no error** — it fails *silently*.
- **Hardware transforms** in Media Foundation interact badly with virtual
  devices.
- DroidCam registers **multiple confusingly named devices** ("DroidCam Source
  1/2/3"), and which one (if any) carries video is unpredictable.
- Behaviour is **machine-dependent** — worked on one PC, frozen on two others,
  with the same app versions.

**Design rule for the clone: do not build or depend on a virtual-camera driver
at all.** Stream over the network (HTTP) and let OpenCV read the URL directly.
This is the same mechanism that makes "IP Webcam"-style apps and DroidCam's own
Wi-Fi mode reliable.

---

## 5. Recommended architecture (no virtual camera anywhere)

### High level

```
┌─────────────────────────── Android phone ───────────────────────────┐
│  CameraX  ──►  per-frame JPEG encode  ──►  embedded HTTP server      │
│                                            serving an MJPEG stream    │
│                                            at  /video  on port 8080   │
└───────────────────────────────────────────────────────────────────────┘
                 │                                  │
        USB (primary)                        Wi-Fi (fallback)
   adb forward tcp:8080 tcp:8080          phone & PC on same network
                 │                                  │
                 ▼                                  ▼
   PC: cv2.VideoCapture(                PC: cv2.VideoCapture(
       "http://localhost:8080/video")      "http://<phone-ip>:8080/video")
```

### MJPEG over HTTP — the format

Serve `multipart/x-mixed-replace; boundary=frame`. Each part is a JPEG:

```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg
Content-Length: <n>

<JPEG bytes>
--frame
Content-Type: image/jpeg
...
```

OpenCV, browsers, VLC, and ffmpeg all read this natively. This is the single
most important interoperability decision in the whole app.

### USB transport (primary — lowest latency, no router involved)

The phone's HTTP server listens on `localhost:8080` *on the phone*. Bridge it to
the PC with Android Debug Bridge:

```
adb forward tcp:8080 tcp:8080
```

Now `http://localhost:8080/video` on the **PC** is the phone's stream. This
needs **USB debugging** enabled on the phone (Developer Options) and `adb` on
the PC (ships with Android Platform Tools). No proprietary driver, no virtual
camera. This is robust and reproducible across machines — exactly what DroidCam
failed to be.

### Wi-Fi transport (fallback)

If USB/adb is inconvenient, the same server is reachable at
`http://<phone-ip>:8080/video` when both devices are on the same network. The
app should display its own IP + port on screen.

### PC-side integration (small, future, NOT part of this doc's build)

`game/capture.py`, `find_cam.py`, and `live_bg_test.py` currently take an int
`--device`. They would gain an optional `--url` that, when present, is passed
straight to `cv2.VideoCapture(url)`. One-line core change; everything
downstream (YOLO, homography, game) is unchanged because it all operates on the
decoded `frame` ndarray.

---

## 6. App requirements that specifically prevent the pain we hit

Build these in from day one — each maps to a concrete DroidCam failure:

- [ ] **Configurable resolution + FPS** (e.g. 640×480 / 1280×720 @ 15/30 fps).
      *DroidCam's free tier capped resolution/FPS.*
- [ ] **One clearly-labelled endpoint** (`/video`). *DroidCam's multiple
      "Source N" devices made it impossible to know which to open.*
- [ ] **Show the exact stream URL + port big on screen** (both the
      `localhost:8080` USB note and the `http://<ip>:8080` Wi-Fi URL).
- [ ] **Foreground service + keep-screen-on / wake lock** so streaming doesn't
      pause when the screen dims. *Phone going idle = frozen stream.*
- [ ] **Adjustable JPEG quality** (e.g. 50–90) to trade latency vs clarity.
- [ ] **A `/snapshot` endpoint** returning a single JPEG, for instant
      connectivity testing in a browser.
- [ ] **Fail loudly, never serve a static placeholder.** If the camera isn't
      streaming, return an HTTP error or close the connection — do **not** emit
      a frozen frame. *The frozen-buffer-with-no-error behaviour is exactly what
      cost us hours; we couldn't tell "broken" from "dark".*
- [ ] **Rear-camera default, with a front/rear toggle.**
- [ ] **Auto-restart the stream on client reconnect.**

---

## 7. Suggested Android tech stack

- **Language**: Kotlin.
- **Camera**: CameraX (`ImageAnalysis` use case to get frames as `ImageProxy`;
  convert YUV→JPEG, or use `ImageCapture`/`YuvToRgbConverter` + `Bitmap`→JPEG).
- **HTTP server**: NanoHTTPD (tiny, battle-tested for exactly this MJPEG
  use-case) or Ktor embedded server if more structure is wanted.
- **Service**: a foreground `Service` holding the camera + server + a wake lock,
  with a persistent notification.
- **USB setup command (document it in-app)**:
  ```
  adb forward tcp:8080 tcp:8080
  ```
- **Permissions**: `CAMERA`, `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_CAMERA`,
  `INTERNET`, `WAKE_LOCK`.

A known-good reference design to study is the open-source **"IP Webcam"** app
behaviour and any **NanoHTTPD MJPEG** example — they do precisely this MJPEG/HTTP
streaming and are reliably readable by OpenCV.

---

## 8. Verification (for whoever builds the app)

Test in increasing order of integration — don't skip to the game:

1. **Browser test (proves the stream exists):** open
   `http://<phone-ip>:8080/video` in a desktop browser on the same Wi-Fi. You
   should see live video. Then `http://<phone-ip>:8080/snapshot` for a single
   frame.
2. **USB test:** run `adb forward tcp:8080 tcp:8080`, then open
   `http://localhost:8080/video` in the PC browser.
3. **OpenCV test (proves it's usable by the project):** the 10-line snippet
   below must show **moving** video and a **changing** mean (the exact check
   that DroidCam failed):

   ```python
   import cv2, numpy as np
   cap = cv2.VideoCapture("http://localhost:8080/video")
   for _ in range(60):
       ok, f = cap.read()
       if ok:
           print(round(float(np.mean(f)), 2), f.shape)
           cv2.imshow("test", f)
           if cv2.waitKey(1) == 27: break
   cap.release(); cv2.destroyAllWindows()
   ```
   - If the printed mean **varies** frame-to-frame → success, the stream is
     live and OpenCV-readable.
   - If it's **constant** (DroidCam's failure mode) → something is wrong; fix it
     before integrating.
4. **Only then** wire it into the game by giving `cv2.VideoCapture` the URL.

---

## 9. TL;DR for the next Claude instance

- Build an **Android** app that streams the rear camera as **MJPEG over HTTP**
  on `:8080/video`, plus a `/snapshot` endpoint.
- Reach the PC via **`adb forward tcp:8080 tcp:8080`** (USB, primary) or the
  phone's IP (Wi-Fi, fallback).
- **Never** build/use a Windows virtual-camera driver — that is the single
  thing that broke DroidCam for us (frozen/green buffer, silent failure,
  machine-dependent).
- **Never serve a frozen placeholder frame** — fail loudly instead.
- The PC side reads it with `cv2.VideoCapture("http://localhost:8080/video")`,
  which "just works" — so the existing game needs only a one-line change to
  accept a URL instead of a device index.
