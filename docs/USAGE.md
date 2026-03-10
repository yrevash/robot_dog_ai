# REVO Robot Dog — Usage Guide

## Requirements

```bash
pip install opencv-contrib-python numpy mediapipe
```

> **Raspberry Pi only:** also run `sudo apt install alsa-utils` for bark audio.

---

## Step 1 — Enroll a Face

Sit in front of the camera in good lighting. Run:

```bash
python face_embedding.py enroll --name YourName --samples 25
```

- A window opens showing your camera feed.
- A **green box** means the face is ready to capture.
- Press **SPACE** to capture each sample (need 25 total).
- Press **Q** to quit early.

After capturing, the database is built automatically (`face_db.npz`).

> Re-enroll or add more samples anytime — just run the same command again (add `--replace` to overwrite existing samples).

---

## Step 2 — Run on Raspberry Pi

```bash
python revo_pi.py
```

That's it. The terminal will show:

```
→ DETECTING       # face seen, building votes
→ AUTHORIZED: YourName   # identity confirmed, gestures active
GESTURE: FORWARD → forward   # command sent to robot
```

### With robot HTTP endpoint

```bash
python revo_pi.py --iot-url http://192.168.x.x:8080/cmd
```

Commands are sent as JSON `POST`:
```json
{"person": "YourName", "command": "forward", "source": "gesture", "timestamp": 1234567890.0}
```

---

## Step 3 — Gesture Commands

Gestures only work **after your face is authorized**. Hold each sign steady for ~0.5–1 sec.

| Sign | Command |
|---|---|
| Open palm (all 5 fingers up) | `forward` |
| Thumb down, fist closed | `backward` |
| Only index up, leaning left | `left` |
| Only index up, leaning right | `right` |
| V sign (index + middle) | `sit` |
| Index + middle + ring up | `stand` |
| 4 fingers up, thumb folded | `walk` |
| Only pinky up | `tail_wag` |
| Fist (all closed) | `stop` |
| Thumb-index pinch + 3 fingers up | `bark` |

> Palm must face the camera. Back-of-hand gestures are ignored.

---

## Config (optional)

Generate a config file to tune thresholds without touching code:

```bash
python revo_pi.py --save-config   # writes revo_config.json
```

Key fields:

| Field | Default | What it does |
|---|---|---|
| `iot_url` | `""` | Robot HTTP endpoint |
| `bark_audio` | `""` | Path to `.wav`/`.mp3` for bark sound |
| `light_normalize` | `false` | Enable CLAHE if lighting is poor |
| `gesture_enabled` | `true` | Turn gestures on/off |
| `frame_skip_idle` | `5` | CPU saving when no face present |

---

## CLI Reference

```bash
python revo_pi.py                          # run normally
python revo_pi.py --enroll Alice           # enroll Alice then run
python revo_pi.py --iot-url http://x/cmd   # set robot URL
python revo_pi.py --no-gesture             # face-only, no gesture
python revo_pi.py --verbose                # debug logging
python revo_pi.py --save-config            # write revo_config.json
```

```bash
# Desktop GUI (non-RPi)
python face_control_center.py

# CLI pipeline steps (manual)
python face_embedding.py capture --name Alice --samples 25
python face_embedding.py build
python face_embedding.py recognize
```
