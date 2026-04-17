# REVO Pi Deploy

Headless Raspberry Pi server for the REVO robot dog.

- Runs face recognition + hand-gesture detection on a USB webcam plugged into the Pi
- Forwards recognised gestures to the ESP32 `walk_test.ino` firmware over **GPIO UART**
- Exposes a mobile-friendly web UI on the local network for live view, manual control, and mode switching

```
Phone / Laptop  ──HTTP──►  Raspberry Pi (FastAPI + AI)  ──UART GPIO──►  ESP32  ──I2C──►  Servos
```

## Three camera / AI modes

Pick any of these from the UI dropdown:

| Mode | Where the camera is | Where AI runs | Notes |
|---|---|---|---|
| `pi_camera` | USB webcam on the Pi | Pi | Default. Fully standalone. |
| `laptop_stream` | Laptop / phone browser (via `getUserMedia`) | Pi | Frames are streamed over WebSocket to the Pi, which runs inference. |
| `laptop_client` | Laptop | Laptop (`laptop_client.py`) | Laptop runs the heavy models locally and only POSTs the recognised gesture to the Pi. Cheapest on the Pi. |

---

## Setup on the Raspberry Pi 5

```bash
# 1. Enable the Pi 5 hardware UART (so /dev/serial0 exists)
sudo raspi-config
#   → Interface Options → Serial Port
#       Would you like a login shell over serial?           No
#       Would you like the serial port hardware enabled?    Yes
sudo reboot

# 2. Clone the repo and enter the deploy folder
git clone <your-repo-url> Revo_Robot_AI
cd Revo_Robot_AI

# 3. Create venv + install Pi-side deps
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r pi_deploy/requirements.txt

# 4. Give your user permission to use the UART without sudo
sudo usermod -a -G dialout "$USER"
# log out and back in so the group takes effect

# 5. Make sure face data & models exist (copied from your laptop or enrolled on Pi)
ls data/face_db.npz                            # must exist
ls data/known_faces/                           # at least one person

# 6. Start the server
./pi_deploy/run.sh
```

Open `http://<pi-ip>:8080` on your phone (same WiFi) and you're in.

### Wiring (Pi GPIO ↔ ESP32 Serial2)

```
Pi pin 8  (GPIO 14, TXD) ──► ESP32 GPIO 16 (RX2)
Pi pin 10 (GPIO 15, RXD) ◄── ESP32 GPIO 17 (TX2)
Pi pin 6  (GND)          ──► ESP32 GND          ← REQUIRED
```

Both sides are 3.3 V logic, no level shifter needed.

### Firmware note

The stock `walk_test.ino` reads from USB `Serial`. For the GPIO link we'll flip
it to `Serial2` on pins 16/17. That's a one-line change — see
`docs/pi_esp32_integration.md` for the protocol; the command set is unchanged.

---

## Running on your laptop (no Pi)

Everything auto-falls back to mock UART when `/dev/serial0` is missing, so the
full app runs on your laptop:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r pi_deploy/requirements.txt
REVO_MOCK_UART=1 ./pi_deploy/run.sh
```

Commands get logged to stdout instead of being written to a real ESP32.

### Laptop AI mode (mode B)

On any machine with a webcam (laptop), run:

```bash
python pi_deploy/laptop_client.py --pi-url http://<pi-ip>:8080 --preview
```

This captures the laptop camera, runs the face + gesture models locally, and
POSTs each recognised gesture to the Pi's `/api/gesture` endpoint.

---

## API (for scripts / curl)

| Method | Path | Body | Purpose |
|---|---|---|---|
| GET  | `/api/state` | — | Current snapshot (mode, authorized, fps, …) |
| POST | `/api/mode` | `{"mode": "pi_camera"}` | Switch mode |
| POST | `/api/command` | `{"cmd": "walk"}` | Raw ESP32 line |
| POST | `/api/gesture` | `{"gesture": "walk", "person": "Alice"}` | Dispatch gesture |
| POST | `/api/speed/{1..5}` | — | Set walking speed |
| POST | `/api/stand` `/walk` `/stop` | — | Shortcuts |
| GET  | `/video_feed` | — | MJPEG of annotated frames |
| WS   | `/ws/status` | — | Pushes RobotState snapshots |
| WS   | `/ws/frames` | binary JPEG | Used by browser in `laptop_stream` mode |

---

## File map

```
pi_deploy/
├── app/
│   ├── main.py            # FastAPI routes + websockets
│   ├── config.py          # All tunables (serial device, thresholds, etc.)
│   ├── uart_bridge.py     # ESP32 walk_test commands as Python functions
│   ├── gesture_map.py     # gesture name → ESP32 command sequence
│   ├── ai_worker.py       # background thread: read frame → face → gesture → UART
│   ├── camera_source.py   # Pi webcam OR remote-pushed frames
│   ├── robot_state.py     # live state snapshot + pub/sub
│   └── ai/
│       ├── face.py        # YuNet + SFace wrapper (reuses src/face_embedding.py)
│       └── gesture.py     # MediaPipe hand-landmark rules
├── web/
│   ├── index.html         # Mobile UI
│   ├── style.css
│   └── app.js
├── docs/
│   ├── GUIDE.md
│   └── pi_esp32_integration.md
├── laptop_client.py       # Mode B runner (laptop does AI, Pi only relays)
├── requirements.txt
├── run.sh
└── README.md
```

---

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `REVO_SERIAL` | `/dev/serial0` | UART device path |
| `REVO_MOCK_UART` | `0` | `1` to force mock mode |
| `REVO_PORT` | `8080` | HTTP port |

---

## Safety

All angle clamping lives in the ESP32 firmware (see `docs/pi_esp32_integration.md`).
The Pi never needs to enforce them — even if the UI sends `lfa 999`, the ESP32
will clamp to the servo's allowed range before writing PWM. One less thing
to worry about when experimenting.
