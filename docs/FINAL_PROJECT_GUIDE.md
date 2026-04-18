# REVO 

## What the project does

A user walks up to a camera. The system figures out **who** they are from their face. Only enrolled people get past this check. Once authorized, they can control a quadruped robot dog with **hand gestures** — open palm to walk, fist to stop, V-sign to sit, and so on.

We've built three things that run together:

1. A **Python backend** on a Raspberry Pi 5 that does the AI (face + gesture) and exposes a REST/WebSocket API.
2. A **web UI** that anyone on the same WiFi can open from their phone or laptop to see the live video feed, check which person is authorized, and manually override the robot.
3. A thin **ESP32 firmware** that receives one-word commands from the Pi over UART and actually drives the servo motors.

That's the shape. The rest of this doc is about the interesting pieces.

---

## Tech stack — what's actually doing the work

| Layer | Tool | Why |
|---|---|---|
| Face detection | **YuNet** (OpenCV DNN, ONNX) | Light, fast, open, works on Pi without a GPU |
| Face recognition | **SFace** (OpenCV DNN, ONNX) | Produces 128-D embeddings; no heavy training |
| Hand landmarking | **MediaPipe HandLandmarker** | 21 hand keypoints at ~30 FPS on CPU, trained model available from Google |
| Gesture classification | Rule-based on MediaPipe landmarks | Interpretable, no data collection needed, zero training |
| Backend | **FastAPI + uvicorn + threading** | Async HTTP + WebSocket, background worker thread |
| Frontend | Plain **HTML/JS** | No build step, runs on any phone browser |
| Pi ↔ ESP32 link | **UART over GPIO (3.3 V, 115200 baud)** | No USB, no PC in the loop at runtime |

Total model weight: ~50 MB for YuNet + SFace + HandLandmarker combined. Everything runs on CPU.

---

## The face recognition pipeline

```
camera frame
    │
    ▼
 YuNet detector  →  face bounding box + quality score
    │
    ▼
 quality gate (area ratio ≥ 0.08, not at frame edge, score ≥ 0.9)
    │
    ▼
 SFace recognizer on aligned crop  →  128-D embedding (L2-normalized)
    │
    ▼
 cosine similarity vs. enrolled database
    │
    ▼
 two-gate matching  →  candidate identity
    │
    ▼
 6-frame voting  →  authorized name / "Unknown"
```

### Enrollment (offline, one-time per person)

For each person we want the system to recognize:

1. Capture ~25 photos of the face at different angles (we have a CLI tool for this: `face_embedding.py capture --name Alice`)
2. Run each photo through YuNet → crop → SFace → 128-D embedding
3. Apply **light normalization** (CLAHE in LAB space + gamma correction) and take 4 augmented versions to handle lighting changes
4. Save all embeddings to `data/face_db.npz`, plus one **centroid** per person (the mean embedding)

### Recognition at runtime

For each incoming frame:

1. **Detect** with YuNet. It gives you a box and a landmark set (5 points: eyes, nose, mouth corners). These are used to align the crop before embedding.
2. **Embed** the aligned crop with SFace → 128-D vector, L2-normalized.
3. **Match** with the **two-gate approach** — this is the interesting bit.

### The two-gate identity matcher

We compute cosine similarity (dot product on L2-normalized vectors, so just a matrix multiply: `db_embeddings @ emb`). This gives one similarity score per enrolled sample.

**Gate 1 — sample-level.** Find the highest-scoring enrolled sample. Call its identity the "candidate". We require:
- Top similarity score ≥ **0.42** (prevents random faces from matching)
- Margin between top identity and second-best identity ≥ **0.06** (prevents ambiguous matches — if Alice and Bob both score 0.65, we reject)

**Gate 2 — centroid-level.** Compute similarity to each person's centroid (the mean of their embeddings). Require:
- Candidate's centroid similarity ≥ **0.40**
- Centroid identity must match the sample-level candidate (agreement check)

If both gates pass, we call it a match. Otherwise, "Unknown".

**Why two gates?** Sample-level catches the exact pose the user showed during enrollment. Centroid-level catches the average representation, filtering out lucky single-sample matches. Requiring both to agree is much stricter than either one alone — our false accept rate on a held-out test set dropped from ~8% (single gate) to under 1% (two gates).

### Temporal voting — the last layer of defense

One frame can be wrong even with the gates. So we keep a **deque of the last 6 recognized identities** and only declare authorization if **at least 4 of the 6 agree** on the same person. This kills single-frame flukes at the cost of ~200ms of lag. Acceptable trade-off.

---

## The gesture recognition pipeline

Face is authorized → now we look at the user's hand.

```
camera frame
    │
    ▼
 MediaPipe HandLandmarker  →  21 landmarks (x, y, z)
    │
    ▼
 palm orientation check  (is the palm facing the camera?)
    │
    ▼
 finger-state classifier   →  (thumb, index, middle, ring, pinky) bools
    │
    ▼
 pattern matching          →  gesture name
    │
    ▼
 hold-time gate (0.5 s)  +  cooldown (0.8 s)
    │
    ▼
 → command to robot
```

### MediaPipe HandLandmarker

Google ships a pre-trained model that gives you 21 keypoints on the hand — each keypoint is `(x, y, z)` in normalized coordinates. `z` is relative depth (negative = closer to camera). The model runs on CPU in ~30ms per frame on a Pi 5.

We use the **Tasks API** (MediaPipe 0.10+), not the older `mp.solutions.hands`. The Tasks API is more stable and ships a pre-compiled `hand_landmarker.task` model file.

### Classification — why rule-based and not ML

You could train an SVM / Random Forest / KNN on the 21 landmarks × coordinates (= 63 features) to classify gestures. We tried this (it's in our experiments folder as `eval_gesture.py`). ML classifiers scored slightly higher on a curated test set (94% vs 91%), but required us to collect a per-user gesture dataset, and they failed in weird ways on out-of-distribution inputs.

We stuck with **rule-based** because:
- Zero data collection — we just describe what "fingers up" means and code it up.
- Interpretable. If a gesture doesn't fire, we can print the finger states and immediately see what's wrong.
- Robust to lighting / skin color / hand size — the rules only look at landmark positions, not pixel appearance.
- The gesture set is small (10 gestures) and visually distinct, so rules handle them fine.

### How the rules work

For each of the 4 finger groups (index, middle, ring, pinky), we check if the **fingertip is above the PIP joint** (by y-coordinate). If yes, finger is "up". For the thumb we check `thumb_tip.x < thumb_mcp.x` because the thumb bends sideways, not up.

We also require the **palm to face the camera**. We use z-coordinates: if the wrist, index-MCP, and pinky-MCP are all at roughly similar depth (small z difference), the hand plane is frontal. If one of them is much closer or farther, the hand is rotated and we reject.

Once we have `(thumb, index, middle, ring, pinky)` as booleans, the patterns are obvious:

| Pattern | Gesture | Robot command |
|---|---|---|
| All 5 up | open palm | `walk` |
| None up | fist | `stop` |
| Index + middle up only | V-sign | `sit` |
| Index + middle + ring up | 3 fingers | `stand` |
| Only pinky up | — | `tail_wag` |
| Only index up | point | `left` / `right` (depending on x-direction) |
| Thumb only | thumb up/down | `backward` |

### Hold-time + cooldown — why gestures don't spam

If we fired the command the instant we classified a gesture, one frame of accidental "open palm" while scratching your face would send the robot walking. To fix that:

- **Hold time (0.5 s):** the current gesture must be the same for at least half a second before we emit it.
- **Cooldown (0.8 s):** after emitting a gesture, we don't re-emit the same one until 0.8 s have passed. Moving from "open palm" to "fist" fires instantly, but holding "open palm" keeps firing at most once per second.

This is just three variables in `gesture.py` — dead simple, very effective.

---

## The orchestration layer

FastAPI server, background worker thread, WebSocket for live status.

### Threads and processes

```
main process (FastAPI + uvicorn, async)
│
├── HTTP handlers           — runs in uvicorn's threadpool for `def` endpoints
│                             (async `async def` would hog the event loop since
│                              our handlers call sync code like serial.write)
│
├── WebSocket handlers       — async, run directly on the event loop
│
└── AI worker thread         — background daemon thread started on FastAPI lifespan
    ├── grab frame from FrameSource
    ├── run face.process(frame)
    ├── if authorized: run gesture.process(frame)
    ├── if gesture: dispatch to UART bridge
    ├── encode annotated frame as JPEG for MJPEG stream
    └── loop (~15 FPS on Pi 5)
```

### The state store (`robot_state.py`)

We keep the current state of the system — `authorized` person, `last_gesture`, `last_command`, `mode`, `fps`, etc. — in a thread-safe dataclass. Any code that wants to change state calls `store.update(...)`.

Crucial detail: `store.update()` **only broadcasts over WebSocket when a value actually changes**. Without this, the AI worker would push 15 identical state snapshots per second to every connected browser, which wastes bandwidth and is annoying.

### Why we used threads, not async all the way

OpenCV, MediaPipe, and `serial.Serial` are all **blocking** APIs. Running them on the asyncio event loop would freeze every other HTTP handler every 70ms. So the AI work happens on a regular Python thread (GIL-released during native ONNX/MediaPipe calls — the CPU scheduling actually works fine), and the event loop handles only HTTP/WS I/O.

### The UART bridge

`uart_bridge.py` wraps `pyserial`. It exposes every ESP32 command as a Python method (`bridge.walk()`, `bridge.sit()`, `bridge.joint("lf", "a", 45)`), with a **thread-safe lock** around the write so two callers can't garble each other's commands on the wire.

If `/dev/serial0` (or `/dev/ttyAMA0` on Ubuntu) doesn't exist, it falls back to a **mock** that logs the command instead of writing bytes. That's why the entire server runs fine on a laptop with no hardware attached — we developed everything in mock mode first.

---

## The three camera / AI modes

The UI has a dropdown. Switching it just changes where the video frames come from and where the AI runs:

| Mode | Camera | AI runs on | Use case |
|---|---|---|---|
| `pi_camera` | USB webcam plugged into Pi | Pi | Fully standalone, Pi does everything |
| `laptop_stream` | Phone/laptop browser via `getUserMedia()` | Pi (receives frames over WS) | Use a nicer camera, Pi still does AI |
| `laptop_client` | Laptop webcam | Laptop (`laptop_client.py`) | Offload the heavy ML from the Pi |

In `laptop_client` mode the Pi doesn't run any ML at all — the laptop does the inference locally, recognizes a gesture, and POSTs it to `/api/gesture` on the Pi. The Pi just forwards the command to the ESP32.

Why three modes? Because the Pi 5 can run the full pipeline at ~15 FPS, but if the Pi is already busy (serving many clients, low power, etc.) we can shift the inference to a stronger machine without changing anything else. The robot control path is identical in all three modes.

---

## The Pi and the ESP32 (software-side)

### Why the Pi runs the AI instead of a PC

- **Low latency.** The webcam, the AI, and the UART are all on the same device. No network round-trip for vision → command.
- **Portable.** Once the robot is untethered, there's no laptop in the loop.
- **Enough compute.** Pi 5 runs YuNet + SFace + MediaPipe at 15 FPS comfortably. Pi 4 struggles (~5 FPS) — that's why we target Pi 5.

### Why the ESP32 runs the motors (not the Pi directly)

- **Hard real-time timing.** Servos need 50 Hz PWM with microsecond-precision pulse widths. Linux on the Pi has jitter measured in milliseconds — unacceptable for smooth gait. The ESP32 talks to a **PCA9685 PWM driver** over I2C, which generates clean hardware PWM.
- **Separation of concerns.** The ESP32 owns motor safety (angle clamping per servo, so Pi can never command a servo past its mechanical limit). The Pi owns intelligence. Clean split.
- **Simpler code on both sides.** The Pi talks a tiny ASCII protocol (`walk\n`, `sit\n`, `lfa 45\n`). No real-time constraints, no motor math.

### The wire between them

3 jumper wires: Pi GPIO 14 (TXD) → ESP32 GPIO 16, Pi GPIO 15 (RXD) ← ESP32 GPIO 17, and a common GND. Both sides 3.3 V logic, 115200 baud, text protocol. Each command terminated by `\n`.

The ESP32 firmware (`revo_uart.ino`) listens on **both** its USB port and the GPIO UART simultaneously. This means we can still debug from the Arduino Serial Monitor while the Pi is driving the robot — invaluable during development.

---

## Decisions we made and why

Things you might get asked in a viva:

**Why YuNet and not MTCNN / Haar cascades?** YuNet is the current state-of-the-art for lightweight face detection on CPU (70% WIDER-Face mAP at 320×320). Haar cascades are 20 years old and miss tilted faces. MTCNN is slower and has no ONNX release maintained by OpenCV.

**Why SFace and not FaceNet / ArcFace?** SFace is trained on the same dataset as ArcFace but ships as a 38 MB ONNX model tuned for real-time CPU inference. FaceNet requires TensorFlow in the pipeline; SFace needs only OpenCV DNN.

**Why cosine similarity and not Euclidean?** L2-normalized embeddings live on the unit sphere. Cosine similarity is equivalent to Euclidean in that space but clamps scores to `[-1, 1]`, which makes threshold tuning easy (we picked 0.42 by sweeping ROC on held-out pairs).

**Why 6-frame voting with 4 required?** We tested (H, V) pairs from (3, 2) to (10, 7). (6, 4) gave the best trade-off: authorization latency under 400 ms on a 15 FPS feed, false-accept rate drops to near zero on our test set.

**Why MediaPipe over OpenPose for hands?** OpenPose is heavy (200+ MB), needs a GPU to hit reasonable FPS. MediaPipe's HandLandmarker is 7 MB, CPU-first, and Google maintains it.

**Why rule-based gestures and not an SVM?** Covered above — zero training data required, interpretable, robust. We benchmark against ML classifiers in `experiments/eval_gesture.py`; ML wins by 2–3 percentage points in-distribution but loses out-of-distribution. The intent of the system is control, not a paper, so we picked the predictable option.

**Why FastAPI and not Flask?** We needed WebSockets (live status broadcast to the UI, live frame pushes from the laptop). FastAPI's async WebSocket support is built in. Flask needs extensions. Also, Pydantic type validation on request bodies is free in FastAPI.

**Why plain HTML/JS and not React?** The UI is four screens of buttons and one video element. A build step would add complexity with no benefit. Every phone browser renders this without any dependencies.

**Why GPIO UART and not USB between Pi and ESP32?** USB through the Pi's root hub works, but if you ever need to unplug the laptop for an on-robot deploy, the Pi is suddenly the USB host — which adds an enumeration step and another failure mode. GPIO UART is always-on, 3 wires, no drivers.

**Why ASCII text and not a binary protocol?** Debuggability. Anyone can `cat /dev/ttyAMA0` or open Arduino Serial Monitor and see exactly what's being sent. Binary framing would save a few bytes per command; at 115200 baud that's imperceptible.

---

## What the system doesn't do yet (and why that's fine)

- **Multi-person arbitration.** If two enrolled people are in frame, we just pick the highest similarity. A proper system would require the authorized user to actively claim control. We skipped it because our use case is single-operator.
- **Gesture sequences.** Each gesture fires immediately after its hold time. We don't chain gestures like "circle + fist → emergency stop". Could be added later with a finite-state machine.
- **Privacy / encryption.** The web UI is plain HTTP on a local network. Trusted-LAN only. Adding HTTPS is one openssl command + two uvicorn flags, but isn't needed for home WiFi use.

---

## How to run it end-to-end

Assuming the Pi is set up (see `pi_deploy/docs/SETUP.md`):

```bash
# On the Pi
cd ~/robot_dog_ai
source venv/bin/activate
./pi_deploy/run.sh
```

Open `http://<pi-ip>:8080` from any browser on the same WiFi. Press **Start AI**. Show your face. Show an open palm.

If you want the laptop to do the AI (Pi under load or without proper PSU):

```bash
# On the laptop
cd /path/to/Revo_Robot_AI
source venv/bin/activate
python pi_deploy/laptop_client.py --pi-url http://<pi-ip>:8080 --preview
```

That's the whole system in two commands.

---

## Where the code lives (one-paragraph tour)

- `src/face_embedding.py` — the original face pipeline, everything we know about detection + recognition lives here. `FaceRecognizer` in `pi_deploy` is a thin wrapper around this.
- `pi_deploy/app/ai/gesture.py` — MediaPipe + finger-state rules.
- `pi_deploy/app/ai_worker.py` — the thread that ties face + gesture + UART together.
- `pi_deploy/app/main.py` — FastAPI routes.
- `pi_deploy/app/uart_bridge.py` — serial writer with mock fallback.
- `pi_deploy/web/` — the UI, three files of vanilla HTML/CSS/JS.
- `rbdg/tools/revo_uart/revo_uart.ino` — the ESP32 firmware. Listens on USB + GPIO UART, clamps every angle to the per-servo safety limit before writing PWM.

