# Duck Brain: curiosity prototype (step 1)

A static web page that runs entirely in the Pixel's browser. No server-side code, no cloud, no PC at runtime.
Everything (libraries, wasm, models, ~31 MB) is in this folder, so it also works offline once served.

## What it does
- **Sees**: objects (MediaPipe EfficientDet-Lite0, 80 COCO classes) and faces (face-api TinyFaceDetector).
- **Fingerprints**: objects get a 1024-number MobileNetV3 vector; faces get a 128-number FaceRecognitionNet vector.
- **Remembers**: each thing becomes a "proto" in IndexedDB with a few fingerprints from different angles, a visit count, smiles, and a thumbnail.
- **Novelty**: far from everything in memory = new; matched = less novel the more visits it has; +bonus if not seen for a day.
- **Boredom**: staring at something makes interest decay (τ ≈ 5 s), so it moves on.
- **Gaze**: picks the most interesting thing (with hysteresis), shows a pink reticle + duck eye, and builds the command the robot would get (`head.yaw/pitch`, `walk.vx/wz`, `mode`).
- **Owner**: "Enroll my face" takes 16 face samples; afterwards the duck labels you ★ and follows you when nothing new is around.

## Run it

Camera access needs **HTTPS or localhost**. That's a browser rule, not this code.

### A. On the phone via GitHub Pages (recommended, free, permanent URL)
1. On github.com create a public repo, e.g. `duck-brain`.
2. From CachyOS:
   ```bash
   cp -r duck-brain ~/duck-brain && cd ~/duck-brain
   git init -b main && git add . && git commit -m "duck brain v1"
   git remote add origin git@github.com:JUST-F1ZZY/duck-brain.git
   git push -u origin main
   ```
   (Biggest file is the 14.6 MB wasm, under GitHub's 100 MB limit.)
3. Repo → Settings → Pages → Source: *Deploy from a branch*, `main` / `/ (root)`. Wait ~1 min.
4. On the Pixel, open `https://just-f1zzy.github.io/duck-brain/` in Chrome. Pick **Phone camera (back)** or **(front)**, press Start, allow the camera.

### B. Quick test on the PC (localhost counts as secure)
```bash
cd duck-brain && python -m http.server 8000
# open http://localhost:8000 in Chrome/Firefox with a webcam
```

### C. Phone → PC over Wi-Fi without HTTPS
`http://192.168.x.x:8000` is not secure, so the camera is blocked. Workaround: on the phone open
`chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://192.168.x.x:8000`, relaunch Chrome.

## Following Hunter (face and/or body)
- **Face** = long-term identity (enroll once).
- **Body** = today's outfit. Whenever your face is recognised inside a person box, the duck saves a fingerprint of your torso
  (MobileNet shape vector + colour histogram). With your back turned, a person whose torso matches (≥ 0.82) is you, and it follows.
  Once a person box was linked to you by face, it stays you while tracked, even if you turn around.
- Outfit memory expires after 12 h (clothes change). After enrolling, stand back for a few seconds so it sees your whole body.
- Limits: someone in a very similar outfit can be mistaken for you; a jacket on/off breaks the match until it sees your face again.
  In a fake-camera test the same outfit seen without a face scored 0.84, a different person 0.21.

## Using it
1. Start → point at stuff. Green boxes = objects, blue = faces, yellow = owner. `NEW` = not in memory.
2. Front camera at yourself → **Enroll my face** → slowly turn your head for ~5 s.
3. Memory panel: rename/forget things, Export/Import JSON (memory is per-site, so export before switching from GitHub Pages to the Pi-served page).
4. Tuning: if two different mugs merge into one memory, raise *Object match similarity*; if the same mug keeps showing as new, lower it. Same idea for faces with *Face match distance* (lower = stricter).
5. `?cpu` at the end of the URL forces the CPU path if WebGL acts up.

## Swapping in the duck's camera later
Use the source dropdown: **MJPEG stream URL** or **JPEG snapshot URL**. The browser only lets the page read pixels
from another host if that host sends CORS headers, and an HTTPS page can't load `http://` streams or open `ws://`.
Cleanest setup: **the Pi serves this same folder** plus `/stream.mjpg` and a WebSocket on the hotspot. Then everything is
one origin and nothing gets blocked. (The camera on the phone isn't needed in that mode, so the HTTPS rule doesn't matter.)

## Files
| file | role |
|---|---|
| `js/video-source.js` | camera / MJPEG / snapshot / video, one interface |
| `js/vision.js` | the three models |
| `js/tracker.js` | links detections across frames (IoU) |
| `js/memory.js` | IndexedDB memory, matching, export/import |
| `js/curiosity.js` | novelty, boredom, gaze, mode, robot command |
| `js/hooks.js` | WebSocket command sink + stubs for affinity, event memory, contingency |
| `js/main.js` | loop + UI |

## Known limits (honest list)
- Objects are limited to COCO's 80 classes. A weird gadget that isn't one of them won't get a box (the scene-change signal still notices it).
- MobileNet fingerprints see *appearance*: two identical white mugs = one memory, and one mug in very different light might be two.
- Face recognition is solid at 0.5 distance for frontal faces, weaker in profile or dim light.
- Tested in headless Chromium with a fake camera (fps there is meaningless). Real Pixel fps is still to be measured; target 8-12.
