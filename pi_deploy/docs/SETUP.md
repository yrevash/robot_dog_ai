# REVO Pi Setup — step by step

Tight version. Do these in order.

## 0. One-time: flash the ESP32 (from your laptop)

1. Open `rbdg/tools/revo_uart/revo_uart.ino` in Arduino IDE
2. Board: **ESP32 Dev Module**
3. Install library: **Adafruit PWM Servo Driver Library**
4. Upload
5. Serial Monitor @ 115200 → type `ping` → expect `pong`. Type `stand`, `walk`, `sit`. Verify moves.

Firmware listens on USB **and** GPIO 16/17 simultaneously, so you can keep debugging over USB after the wiring is done.

## 1. Wiring

```
Pi pin 8  (GPIO 14, TXD) ──► ESP32 GPIO 16 (RX2)
Pi pin 10 (GPIO 15, RXD) ◄── ESP32 GPIO 17 (TX2)
Pi pin 6  (GND)          ──► ESP32 GND           ← required
```

Both sides are 3.3 V — no level shifter.

## 2. Pi one-time config

```bash
sudo raspi-config
```

- `Interface Options → Serial Port`
  - login shell over serial? → **No**
  - serial port hardware enabled? → **Yes**

```bash
sudo usermod -a -G dialout $USER
sudo reboot
```

SSH still works after this. Only the serial-console login is disabled.

## 3. Clone + install

```bash
git clone <your-repo-url> Revo_Robot_AI
cd Revo_Robot_AI
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r pi_deploy/requirements.txt
```

## 4. Face data

Either copy from your laptop:

```bash
# On laptop
scp data/face_db.npz pi@<pi-ip>:~/Revo_Robot_AI/data/
scp -r data/known_faces pi@<pi-ip>:~/Revo_Robot_AI/data/
```

Or enroll fresh on the Pi (USB webcam plugged in):

```bash
python src/face_embedding.py enroll --name Alice --samples 25
```

## 5. Plug in USB webcam

```bash
ls /dev/video*     # /dev/video0 should appear
```

## 6. Run

```bash
./pi_deploy/run.sh
hostname -I        # note the Pi IP
```

Open `http://<pi-ip>:8080` on your phone (same WiFi) → **Start AI**.

## Verify end-to-end

| Check | Expected |
|---|---|
| Web UI loads | Yes, connection dot goes green |
| UART field | `connected` (not `mock`) |
| Start AI → show your face | Status panel shows your name as `Authorized` |
| Hold open palm | `Last gesture: forward`, `Last cmd: walk`, robot starts trotting |
| Fist | `stop`, robot halts |
| Raw command `ping` | Robot responds `pong` in the log |

## Troubleshooting

| Symptom | Fix |
|---|---|
| UART stays `mock` | `/dev/serial0` missing → redo step 2 and reboot |
| `Permission denied` on serial | Forgot dialout group → re-login or reboot |
| `Database not found` | face_db.npz isn't at `data/face_db.npz` — copy or enroll |
| Camera won't open | Check `ls /dev/video*`; try different USB port |
| Robot doesn't move | Open Serial Monitor on USB, send `ping` — if no response, reflash `revo_uart.ino` |
| Servo twitches | `stop` first, then verify GND is tied between battery, buck, ESP32, and PCA9685 |
