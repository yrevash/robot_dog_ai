# Session notes — REVO Pi deploy + ESP32 UART firmware

Context dump for the next Claude session. What was built, why, and where everything lives.

## What was built

Two halves of the same system:

1. **`pi_deploy/`** — a self-contained FastAPI server that lives inside the
   existing `Revo_Robot_AI` repo. It reuses `src/face_embedding.py` for the
   two-gate face recognition (no duplication) and ships a mobile web UI with
   three camera modes. It talks to the ESP32 over Pi GPIO UART (not USB).

2. **`rbdg/tools/revo_uart/revo_uart.ino`** — a new Arduino sketch that is
   `walk_test.ino` + GPIO Serial2 listening + high-level gesture commands.
   Lives in a sibling folder to `walk_test/` (Arduino convention). `walk_test.ino`
   itself is untouched so it still works as the baseline.

## Why GPIO UART instead of USB

User was worried about leaving USB plugged in during live operation. GPIO 16/17
on the ESP32 are free (GPIO 21/22 are busy with PCA9685 I2C). Pi GPIO 14/15
are UART0. Both sides are 3.3 V — direct connection, no level shifter.

The firmware still listens on USB too, so Arduino IDE debug is not lost.

## Why three camera modes

| Mode | Camera | AI runs on | Use case |
|---|---|---|---|
| `pi_camera` | USB webcam on Pi | Pi | Standalone; only needs the Pi |
| `laptop_stream` | Phone/laptop browser | Pi | Use a nicer camera but keep Pi as brain |
| `laptop_client` | Laptop webcam | Laptop (`laptop_client.py`) | Pi only relays — offloads the heavy models |

Mode is swappable live from the UI dropdown. `set_mode()` now defers opening
the webcam until the AI worker is actually running, so switching modes while
AI is off doesn't grab the camera.

## File map

```
Revo_Robot_AI/
├── pi_deploy/
│   ├── app/
│   │   ├── main.py           FastAPI routes + websockets
│   │   ├── config.py         settings (serial dev, thresholds, ports)
│   │   ├── uart_bridge.py    ESP32 command wrapper; mock fallback when /dev/serial0 missing
│   │   ├── gesture_map.py    gesture name → one-line ESP32 command
│   │   ├── ai_worker.py      background thread: frame → face → gesture → UART → MJPEG
│   │   ├── camera_source.py  PiCameraSource + RemoteFrameSource
│   │   ├── robot_state.py    live state; broadcasts on WS only when value changes
│   │   └── ai/
│   │       ├── face.py       YuNet+SFace wrapper (imports src/face_embedding.py)
│   │       └── gesture.py    MediaPipe hand-landmark rules
│   ├── web/                  vanilla HTML/JS mobile UI — no build step
│   ├── docs/
│   │   ├── GUIDE.md                    project-wide file guide (copy of top-level)
│   │   ├── pi_esp32_integration.md     hardware wiring reference (copy from rbdg)
│   │   ├── SETUP.md                    tight step-by-step for Pi deployment
│   │   └── SESSION_NOTES.md            this file
│   ├── laptop_client.py      mode-B runner (laptop does AI, Pi relays)
│   ├── requirements.txt
│   ├── run.sh
│   └── README.md
└── ...

rbdg/tools/
├── walk_test/walk_test.ino           UNCHANGED — original sketch
└── revo_uart/revo_uart.ino           NEW — walk_test + Serial2 + gesture cmds
```

## Firmware command protocol (revo_uart.ino)

ASCII lines, `\n` terminated, 115200 8N1. Works on both USB Serial and Serial2
(GPIO 16/17). Response goes back on whichever port asked.

Preserved from `walk_test.ino`:
- `stand` / `walk` / `stop` / `info`
- `1` `2` `3` `4` `5` (speed)
- `lfa 45` per-joint, `lf 90` per-leg, `90` all-servos

Added:
- `sit`, `tail_wag`, `bark`, `greet`, `left`, `right`, `backward`
- `ping` → `pong` (link health check)

All angles go through `SMIN/SMAX` clamp before hitting the PCA9685, same as
walk_test. The Pi can send anything — firmware is responsible for not
burning out servos.

## Gesture → wire-command mapping

Because the firmware now owns the composite poses, each gesture is exactly
one line on the wire. `pi_deploy/app/gesture_map.py`:

```
forward / walk  → walk
stop            → stop
stand           → stand
sit             → sit
tail_wag        → tail_wag
bark            → bark
greet           → greet
left            → left
right           → right
backward        → backward
```

## Bugs fixed during the audit pass

| # | Bug | Fix |
|---|---|---|
| 1 | `store.update()` fired WS broadcast every frame | Only notify when any value actually changed |
| 2 | Race in `AiWorker.start()` / `.stop()` | Guarded with `_life_lock`; `_thread` nulled after join |
| 3 | Unhandled exception in `_run()` silently killed the worker | `_iterate()` try/except + 0.1s backoff |
| 4 | `set_mode()` opened the webcam even when AI was stopped | Defers opening until worker is running |
| 5 | `walking` flag never flipped from gesture dispatch | `walking_effect(gesture)` helper merged into `store.update` in both callers |
| 6 | `__import__("time").time()` in route handlers | Plain `import time` at top |
| 7 | MJPEG generator imported cv2/numpy/time inside the loop | Hoisted + cached placeholder JPEG |
| 8 | Face vote history carried across Stop → Start AI | `self._face.reset()` inside `start()` |
| 9 | AI toggle button double-click during slow first-start | `aiPending` guard; button disables, shows `Starting…` |
| 10 | `ai_enabled` defaulted `True` so UI showed `Stop AI` before worker booted | Default flipped to `False` |

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `REVO_SERIAL` | `/dev/serial0` | UART device path |
| `REVO_MOCK_UART` | `0` | `1` forces mock mode even if device exists |
| `REVO_NO_AI` | `0` | `1` makes the server boot the UI without starting the AI worker |
| `REVO_PORT` | `8080` | HTTP port |

## Open items (not done yet)

- **Hardware not connected yet** — user plans to flash firmware + wire GPIO tomorrow. Software has been fully tested end-to-end with mock UART on laptop.
- **Gesture classifier** uses simple rule-based finger states; can't distinguish thumb-up vs thumb-down (both hit `backward`). Acceptable for current gesture set.
- **Face model auto-download** pulls ~50 MB from GitHub on first AI start if `models/` is empty — no progress indicator in the UI.
- **No auth on the web UI.** Trusted-LAN only. User explicitly said no ngrok / no internet-wide access.

## Verified working

- FastAPI server boots on laptop with `REVO_MOCK_UART=1 REVO_NO_AI=1 ./pi_deploy/run.sh`
- `POST /api/gesture {walk}` → `store.walking=true`
- `POST /api/gesture {stop}` → `store.walking=false`
- `store.update` no-op calls produce zero WS broadcasts
- `set_mode()` on a stopped worker does not grab the webcam
- UART bridge mock fallback + real-path imports (`pyserial`) both clean
