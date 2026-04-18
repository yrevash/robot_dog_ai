# REVO deployment journal — 2026-04-18

The real-world walk-through of getting `pi_deploy/` running on Ubuntu 24.04
on a Raspberry Pi 5, with every gotcha encountered and how it was fixed.
Use this alongside `SETUP.md` (the clean version).

Target hardware:
- Raspberry Pi 5 8 GB
- Ubuntu 24.04 LTS (arm64)
- USB webcam (UVC)
- ESP32 Dev Module (Adafruit) running `rbdg/tools/revo_uart/revo_uart.ino`
- Pi 5 + ESP32 connected via 3.3 V GPIO UART (no USB data link at runtime)

---

## What worked end-to-end

Verified live at the end of the session:

1. **UART ping → pong** from Pi to ESP32 over GPIO 14/15 ↔ GPIO 16/17.
2. **Web UI** served at `http://<pi-ip>:8080` reachable from phone + laptop
   on the same WiFi.
3. **Laptop-AI mode** (`laptop_client.py`): MacBook webcam → face recognition +
   gesture detection running locally → POST to Pi `/api/gesture` → Pi relays
   via UART to ESP32. Tested with 7 enrolled identities (Aramaan, Azhar,
   Harshad, Pratham, Shubham, Sohail, Yash). Gestures `forward`, `backward`,
   `sit`, `walk`, `stop` all recognized and round-tripped to the ESP32.
4. **ESP32 firmware** accepts commands on both USB and GPIO Serial2 — the
   same command processor routes replies back to whichever port asked.
5. **Pi-side camera mode** deferred: the Pi 5 power supply we're using
   (MacBook charger, not official 27 W) triggers the Pi 5's USB current cap,
   so the webcam can't be opened on the Pi without a proper PSU or
   `usb_max_current_enable=1` override. Laptop-AI mode bypasses this.

---

## Hardware setup as tested

```
+----------------------+            +----------------+          +-----------+
|  USB-C charger       |            |  Laptop USB    |          |  Battery  |
+----------+-----------+            +--------+-------+          +-----+-----+
           |                                 |                        |
           v                                 v                        v
      +----+-----+      GPIO UART      +-----+-----+              +--+----+
      |  Pi 5    |<------------------->|  ESP32    |              | Buck  |
      |          | 14↔16 / 15↔17 / GND |  revo_uart|              | 5-6V  |
      +----+-----+                     +-----+-----+              +--+----+
           |                                 |                        |
           | (USB cable, webcam)             | I2C                    |
           v                                 v                        v
     +-----+------+                     +----+-----+                  |
     | USB webcam |                     | PCA9685  |<-----------------+
     +------------+                     +----+-----+
                                             |
                                             v
                                      12 × MG90S servos
```

All grounds tied together: battery −, buck OUT−, PCA9685 GND, ESP32 GND,
Pi GND. This is the single most important wire after signal TX/RX.

---

## Step-by-step log (what was actually done, in order)

### 1. Ubuntu-on-Pi-5 serial config

`raspi-config` does not exist on Ubuntu. Instead:

```bash
sudo nano /boot/firmware/config.txt
# appended at the bottom:
#   enable_uart=1
#   dtparam=uart0=on

sudo nano /boot/firmware/cmdline.txt
# checked for console=serial0,... or console=ttyAMA0,... → not present, left alone
# (console=tty1 is HDMI-console, NOT serial — leave it)

sudo systemctl disable serial-getty@ttyS0.service
sudo systemctl disable serial-getty@ttyAMA0.service
sudo usermod -a -G dialout,video $USER
sudo reboot
```

After reboot, `/dev/ttyAMA0` and `/dev/ttyAMA10` both showed up. `ttyAMA0`
is the GPIO 14/15 UART — that's our target. `/dev/serial0` does NOT exist
on Ubuntu (it's a Raspberry Pi OS convention). `run.sh` was updated to
auto-detect both names.

### 2. XHC USB error spam during boot

`XHC-cmd err: 4 type 11 ...` spammed the console on every boot while a
USB Bluetooth dongle was plugged in. Cosmetic — Pi still boots. Moved the
dongle to a USB 2.0 port (black) and the noise disappeared; alternatively
ignore it.

### 3. Clone + venv + deps

```bash
git clone <repo-url> ~/robot_dog_ai
cd ~/robot_dog_ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r pi_deploy/requirements.txt
```

On Ubuntu 24.04 Pi 5 with Python 3.12, all wheels installed cleanly, no
compilation needed. Most were already cached from a previous run.

### 4. Face data sync

From the laptop:
```bash
scp data/face_db.npz yrevashpi@<pi-ip>:~/robot_dog_ai/data/
scp -r data/known_faces yrevashpi@<pi-ip>:~/robot_dog_ai/data/
```

ONNX models auto-download on first `FaceRecognizer` instantiation — no
manual copy needed.

### 5. SSH + firewall

SSH was enabled by default on Ubuntu. Port 22 was already allowed in ufw.
Port 8080 was NOT — had to add:

```bash
sudo ufw allow 8080/tcp
```

### 6. First run — mock UART mode

```bash
./pi_deploy/run.sh
# [run.sh] /dev/serial0 not found — running with REVO_MOCK_UART=1
```

This was the trigger for the `run.sh` rewrite. New version auto-picks
`/dev/ttyAMA0` when `/dev/serial0` is missing. Log line now reads:

```
[run.sh] REVO_SERIAL=/dev/ttyAMA0  REVO_MOCK_UART=0
UART opened: /dev/ttyAMA0 @ 115200
```

### 7. Pi 5 + weak PSU + USB webcam → brown-out

Pi 5 with a MacBook charger (non-official PSU) caps total USB current at
600 mA. A USB webcam pulls ~500–700 mA when streaming → total > cap →
board browns out and shuts off the moment AI worker opens the camera.

Two fixes:
- **Proper fix:** buy the official 27 W USB-C PSU.
- **Override:** add `usb_max_current_enable=1` to `/boot/firmware/config.txt`,
  reboot. Only do this if you trust the PSU can actually deliver the
  current (most 30 W+ USB-C laptop chargers can).

For this session we chose a **third path**: skip the Pi webcam entirely
and use `laptop_client.py` (mode B). The laptop runs the heavy inference,
only POSTs gestures to the Pi.

### 8. Camera permission denied (`/dev/video*`)

After plugging the webcam, `ls /dev/video*` showed a bunch of devices,
but OpenCV got `Permission denied` on all of them. Ubuntu requires the
`video` group. Fixed with `sudo usermod -a -G video $USER` + reboot
(already included in step 1 now).

### 9. MacBook webcam at wrong resolution

`laptop_client.py` originally used `cv2.VideoCapture(0)` without setting
width/height. MacBook default is 1280×720 or higher — YuNet detection was
flaky at that size with the default 0.9 confidence threshold. Patched to
force 640×480:

```python
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

After this change, face detection jumped from ~0 to ~93–94 % confidence
per frame. Also added a 2 s heartbeat log
(`[hb] frames=N faces_seen=M authorized=...`) and red/green bounding box
overlays so the operator can see what the model sees.

### 10. Firmware flash + GPIO link test

1. Opened `~/rbdg/tools/revo_uart/revo_uart.ino` in Arduino IDE
2. Board: ESP32 Dev Module
3. Installed library: Adafruit PWM Servo Driver Library
4. Upload via USB
5. Serial Monitor @ 115200 — sent `ping` → got `pong`. Sent `stand`,
   `walk`, `stop` — firmware responded on USB. Confirmed firmware works
   via USB.
6. Powered ESP32 off, wired three jumpers (GND first), powered back on.
7. Ran the `ping` test from the Pi:

```bash
python3 -c "
import serial, time
s = serial.Serial('/dev/ttyAMA0', 115200, timeout=2); time.sleep(2)
s.write(b'ping\n'); print(s.readline().decode().strip())"
# → pong
```

Link confirmed end-to-end.

### 11. Starting servos (deferred)

Buck output verified at 5–6 V on a multimeter before enabling. First
command sent must be `stand` (smooth 30-step interpolation). Firmware
clamps every angle to `SMIN/SMAX` before writing to PCA9685 — same
safety net as `walk_test.ino`.

---

## Files changed during this deployment session

| File | Change |
|---|---|
| `pi_deploy/run.sh` | Auto-detect `/dev/serial0` → `/dev/ttyAMA0` → mock, log which device it picked |
| `pi_deploy/laptop_client.py` | Force 640×480 capture; 2 s heartbeat log; draw bounding box in preview |
| `CLAUDE.md` | Added Ubuntu-on-Pi-5 path, USB-current cap note, end-to-end test sequence |
| `pi_deploy/docs/DEPLOYMENT_JOURNAL.md` | This file |

No behavior change in `app/main.py`, `app/ai_worker.py`, `app/uart_bridge.py`,
etc. — all the code written yesterday worked on Ubuntu/Pi 5 unchanged.

---

## What's next (for the following session)

- [ ] Power the servos from the buck and test `Stand` → `Walk` → `Stop`
  from the UI. Verify motion matches `walk_test.ino`.
- [ ] Order the official Pi 5 27 W PSU so we can do `pi_camera` mode
  (standalone, no laptop in the loop).
- [ ] Unified battery layout: single 3S LiPo → separate bucks for Pi 5
  (5 V 5 A) and servos (5–6 V 8 A). All grounds tied at battery negative.
- [ ] On-robot mounting + cable management.
- [ ] Optional: add `/api/ping` endpoint that round-trips UART, so the UI
  can self-diagnose the link.

---

## Quick reference — environment variables

| Var | Default | Purpose |
|---|---|---|
| `REVO_SERIAL` | auto-picked by `run.sh` | Force a specific UART device path |
| `REVO_MOCK_UART` | `0` | `1` forces mock even if a device exists |
| `REVO_NO_AI` | `0` | `1` boots the UI without starting the AI worker (no webcam grab) |
| `REVO_PORT` | `8080` | HTTP port |

## Quick reference — one-liner health checks

```bash
# is the UART device there?
ls -l /dev/ttyAMA0

# am I in the right groups?
groups   # should contain "dialout" and "video"

# is ufw blocking 8080?
sudo ufw status

# can I talk to the ESP32?
python3 -c "import serial, time; s=serial.Serial('/dev/ttyAMA0',115200,timeout=2); time.sleep(2); s.write(b'ping\n'); print(s.readline().decode().strip())"

# is the server listening?
ss -tlnp | grep 8080

# can I reach it from laptop?
curl -v http://<pi-ip>:8080/api/state
```
