# Raspberry Pi ↔ ESP32 Integration Guide

Complete context for integrating a Raspberry Pi (running AI/vision) with the ESP32 robot dog. Everything in this file is verified against the actual `tools/walk_test/walk_test.ino` code.

## Integration Architecture

```
┌──────────────────────┐     USB Serial     ┌────────────────┐    I2C    ┌─────────────┐
│  Raspberry Pi        │      115200 baud   │   ESP32        │  (21/22)  │  PCA9685    │
│  - Camera            │ ◄─────────────────►│   walk_test    │  50 Hz    │  16-ch PWM  │
│  - AI inference      │   text commands    │   firmware     │◄─────────►│  0x40       │
│  - Decision logic    │                    │                │           │             │
└──────────────────────┘                    └────────────────┘           └──────┬──────┘
                                                                                │
                                                                                ▼
                                                                         ┌─────────────┐
                                                                         │  12 × MG90S │
                                                                         │   servos    │
                                                                         └─────────────┘
```

- **Pi talks to ESP32 only via USB serial** (the same port used for programming)
- **ESP32 talks to PCA9685 via I2C**
- **PCA9685 generates PWM for all 12 servos**

## Hardware Wiring (VERIFIED)

### ESP32 Pins

| ESP32 Pin | Connects To | Purpose |
|-----------|-------------|---------|
| GPIO 21 (SDA) | PCA9685 SDA | I2C data |
| GPIO 22 (SCL) | PCA9685 SCL | I2C clock |
| VIN | XH-M401 OUT+ | 5V logic power |
| GND | XH-M401 OUT− + PCA9685 GND | Common ground |
| USB | Pi USB port (or laptop) | Serial + programming |

### PCA9685 Board

| PCA9685 Pin | Connects To | Purpose |
|-------------|-------------|---------|
| SDA | ESP32 GPIO 21 | I2C data |
| SCL | ESP32 GPIO 22 | I2C clock |
| VCC | XH-M401 OUT+ | Logic power (PCA chip) |
| GND | Common GND | Ground |
| V+ (screw terminal) | XH-M401 OUT+ | Servo power (high current) |
| GND (screw terminal) | XH-M401 OUT− | Servo ground |

### Power Flow

```
3S LiPo 11.1V 2200mAh 30C
        │
        ▼
  XH-M401 (8A buck, set output ~6V)
        │
        ├──► ESP32 VIN (5-6V ok)
        ├──► PCA9685 VCC (logic)
        └──► PCA9685 V+ (servo power, up to 8A burst)
```

All GNDs tied together (battery −, XH-M401 OUT−, ESP32 GND, PCA9685 GND).

### Servo Channel Map (12 servos on PCA9685 channels 0-11)

| Channel | Leg | Joint | Function |
|---------|-----|-------|----------|
| 0 | LF (Left Front) | Alpha | Hip sideways |
| 1 | LF | Beta | Upper leg forward/back |
| 2 | LF | Gamma | Lower leg (knee) |
| 3 | LB (Left Back) | Alpha | Hip sideways |
| 4 | LB | Beta | Upper leg |
| 5 | LB | Gamma | Lower leg |
| 6 | RF (Right Front) | Alpha | Hip sideways |
| 7 | RF | Beta | Upper leg |
| 8 | RF | Gamma | Lower leg |
| 9 | RB (Right Back) | Alpha | Hip sideways |
| 10 | RB | Beta | Upper leg |
| 11 | RB | Gamma | Lower leg |

Channels 12-15 are unused.

## Communication Protocol (Pi → ESP32)

### Transport

- **Physical**: USB cable from Pi to ESP32
- **Pi device**: `/dev/ttyUSB0` or `/dev/ttyACM0` (run `ls /dev/tty*` after plugging in)
- **Baud rate**: 115200
- **Data format**: ASCII text
- **Line ending**: `\n` (newline) terminates each command

### How walk_test.ino Receives Commands

From walk_test.ino lines 372-375:
```cpp
if (Serial.available()) {
  String input = Serial.readStringUntil('\n');
  input.trim();
  ...
}
```

Every command the Pi sends must end with `\n`. The ESP32 reads until newline, trims whitespace, then matches against the command list.

### Complete Command Reference

| Command | Action | Effect |
|---------|--------|--------|
| `stand\n` | Go to standing pose | Interpolates all 12 servos to calibrated standing angles |
| `walk\n` | Start walking | Continuously executes trotting gait until `stop` |
| `stop\n` | Stop walking | Halts gait, holds current position |
| `1\n` | Set speed 1 (Very Slow) | delay=35ms lift=15° step=12° |
| `2\n` | Set speed 2 (Slow) | delay=25ms lift=20° step=15° |
| `3\n` | Set speed 3 (Medium, default) | delay=15ms lift=25° step=20° |
| `4\n` | Set speed 4 (Fast) | delay=10ms lift=28° step=25° |
| `5\n` | Set speed 5 (Very Fast) | delay=6ms lift=30° step=30° |
| `info\n` | Print angles + speed | ESP32 sends status text back on serial |
| `90\n` | All 12 servos to 90° | Any number 0-180 works (clamped to safety limits) |
| `lf 45\n` | Left Front whole leg to 45° | All 3 LF joints set to 45° (clamped) |
| `lb 90\n` | Left Back whole leg to 90° | |
| `rf 60\n` | Right Front whole leg to 60° | |
| `rb 70\n` | Right Back whole leg to 70° | |
| `lfa 45\n` | LF Alpha only to 45° | |
| `lfb 90\n` | LF Beta only to 90° | |
| `lfg 60\n` | LF Gamma only to 60° | |
| `lba 65\n` | LB Alpha only to 65° | |
| `lbb 40\n` | LB Beta only to 40° | |
| `lbg 60\n` | LB Gamma only to 60° | |
| `rfa 50\n` | RF Alpha only to 50° | |
| `rfb 95\n` | RF Beta only to 95° | |
| `rfg 70\n` | RF Gamma only to 70° | |
| `rba 50\n` | RB Alpha only to 50° | |
| `rbb 95\n` | RB Beta only to 95° | |
| `rbg 80\n` | RB Gamma only to 80° | |

### Safety Limits (ESP32 Enforces Automatically)

Even if Pi sends a value outside these ranges, ESP32 clamps to these before writing to the servo. This prevents burnout (one LB Alpha was lost early on by exceeding limits).

| Channel | Servo | Min° | Max° |
|---------|-------|------|------|
| 0 | LF Alpha | 0 | 90 |
| 1 | LF Beta | 0 | 115 |
| 2 | LF Gamma | 0 | 135 |
| 3 | LB Alpha | 0 | 130 |
| 4 | LB Beta | 0 | 135 |
| 5 | LB Gamma | 0 | 135 |
| 6 | RF Alpha | 10 | 135 |
| 7 | RF Beta | 0 | 135 |
| 8 | RF Gamma | 0 | 135 |
| 9 | RB Alpha | 0 | 90 |
| 10 | RB Beta | 0 | 135 |
| 11 | RB Gamma | 0 | 135 |

### Standing Angles (Reference)

```
LF: Alpha=50°, Beta=60°, Gamma=70°
LB: Alpha=70°, Beta=90°, Gamma=60°
RF: Alpha=50°, Beta=95°, Gamma=70°
RB: Alpha=50°, Beta=95°, Gamma=80°
```

### ESP32 → Pi (Response Messages)

The ESP32 prints status back on serial. Pi can read these or ignore them. Examples:

```
=== Walk + Calibration Test ===
Walk:  stand, walk, stop
Speed: 1(very slow) 2(slow) 3(med) 4(fast) 5(very fast)
...
Standing.
Walking at speed 3
Stopped.
Speed 3 — Medium     (delay=15 lift=25 step=20)
  LF-Alpha -> 50.0°  pulse:275
─────────────────────────────
Speed: 3  delay=15  lift=25  step=20
        Alpha   Beta    Gamma
LF:     50.0°   60.0°   70.0°
LB:     70.0°   90.0°   60.0°
RF:     50.0°   95.0°   70.0°
RB:     50.0°   95.0°   80.0°
─────────────────────────────
```

## How walk_test.ino Works (for AI integration)

### Main Loop Behavior

```cpp
void loop() {
  if (walking) {
    walkCycle();  // one full trot cycle (~1 second at speed 3)
  }
  if (Serial.available()) {
    // parse and execute command
  }
}
```

- When `walking=true`: loop continuously executes `walkCycle()`, which blocks for ~1 sec per cycle
- When `walking=false`: loop only processes serial input
- **Key implication**: Commands sent while walking are queued and processed between walk cycles (max ~1 sec latency in walk mode)

### Walk Cycle Internals

- Trotting gait — diagonal pairs alternate (LF+RB swing, then RF+LB swing)
- Foot trajectory: Cubic Bezier in (beta, gamma) space
- Swing: foot lifts → swings forward → plants down
- Stance: linear beta push-back, gamma stays at standing angle

### Stand Behavior

- On boot: calls `stand()` automatically, robot assumes standing pose
- `stand` command: smoothly interpolates all 12 servos to standing angles over 30 steps

## Pi Integration (Python Example)

### Install dependency
```bash
pip install pyserial
```

### Minimal control script
```python
import serial
import time

# Find correct port: ls /dev/tty* (usually /dev/ttyUSB0 or /dev/ttyACM0)
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # ESP32 resets on serial open, wait for boot

def send(cmd):
    ser.write((cmd + '\n').encode())
    # optional: read response
    # print(ser.readline().decode().strip())

# Stand up
send('stand')

# Set medium speed
send('3')

# Walk for 5 seconds
send('walk')
time.sleep(5)
send('stop')

# Move a specific servo
send('lfa 45')

# Go back to standing
send('stand')

ser.close()
```

### AI-driven flow
```python
while True:
    frame = camera.capture()
    action = ai_model.predict(frame)  # e.g. "walk", "stop", "turn_left"

    if action == "walk":
        send('walk')
    elif action == "stop":
        send('stop')
    elif action == "faster":
        send('4')
    # ... map AI output to commands
```

## Important Caveats

1. **ESP32 resets when Pi opens the serial port** — wait 2 seconds after opening before sending commands
2. **Line ending MUST be `\n`** — without it, ESP32 waits forever for the command to complete
3. **Walking is blocking** — a `walkCycle()` runs for ~1 second before the loop checks for new commands. If low latency is critical, reduce `SWING_STEPS` in walk_test.ino (currently 30 per phase)
4. **Safety limits are hard** — Pi can't override them. Good thing, because this prevents servo burnout.
5. **USB power alone won't run the servos** — the XH-M401 must be powered on, otherwise servos won't move even though PCA9685 receives signals

## Quick Test Checklist (Pi ↔ ESP32)

1. Connect ESP32 to Pi USB
2. Run `ls /dev/tty*` → note the port (e.g. `/dev/ttyUSB0`)
3. Run `sudo chmod 666 /dev/ttyUSB0` (or add user to `dialout` group)
4. Open Python, import serial, open the port at 115200
5. Send `b'info\n'`, read back → should see the angle table
6. Send `b'stand\n'` → robot should move to standing pose
7. Send `b'3\nwalk\n'` → robot should start walking

If these work, the integration is ready for AI commands.
