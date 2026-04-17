"""
UART bridge to the ESP32 running `walk_test.ino`.

Wiring (Pi 5 GPIO UART ↔ ESP32 Serial2):
    Pi  pin 8  (GPIO 14, TXD) ──► ESP32 GPIO 16 (RX2)
    Pi  pin 10 (GPIO 15, RXD) ◄── ESP32 GPIO 17 (TX2)
    Pi  pin 6  (GND)          ──► ESP32 GND

Protocol: ASCII text, 115200 8N1, each command terminated with '\n'.

All safety limits are enforced by the ESP32 firmware — the Pi can send any
angle and the firmware will clamp before writing to the servo.

Every walk_test.ino command is exposed as a Python method. When the serial
device is missing (e.g. running on the laptop), the bridge transparently
falls back to a mock that logs commands instead of writing bytes.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover - pyserial is listed in requirements
    serial = None

log = logging.getLogger(__name__)

LEGS = ("lf", "lb", "rf", "rb")
JOINTS = ("a", "b", "g")  # alpha, beta, gamma
SPEEDS = (1, 2, 3, 4, 5)


class UartBridge:
    """Thread-safe serial writer for the ESP32 walk_test firmware."""

    def __init__(
        self,
        device: str,
        baud: int = 115200,
        timeout: float = 1.0,
        force_mock: bool = False,
        on_send: Optional[Callable[[str], None]] = None,
    ):
        self.device = device
        self.baud = baud
        self.timeout = timeout
        self._lock = threading.Lock()
        self._ser = None
        self._mock = force_mock
        self._on_send = on_send
        self._last_cmd: Optional[str] = None
        self._last_ts: float = 0.0
        self._open()

    # ---------- connection ----------

    def _open(self) -> None:
        if self._mock or serial is None:
            log.warning("UART in MOCK mode (device=%s)", self.device)
            self._mock = True
            return
        try:
            self._ser = serial.Serial(self.device, self.baud, timeout=self.timeout)
            # ESP32 resets on serial open — wait for boot before first command
            time.sleep(2.0)
            log.info("UART opened: %s @ %d", self.device, self.baud)
        except Exception as e:
            log.warning("UART open failed (%s); falling back to MOCK: %s", self.device, e)
            self._ser = None
            self._mock = True

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None

    @property
    def is_mock(self) -> bool:
        return self._mock

    @property
    def last_command(self) -> Optional[str]:
        return self._last_cmd

    # ---------- raw send ----------

    def send(self, cmd: str) -> str:
        """Send a single line to the ESP32. Returns the exact string sent."""
        cmd = cmd.strip()
        if not cmd:
            return ""
        line = cmd + "\n"
        with self._lock:
            self._last_cmd = cmd
            self._last_ts = time.time()
            if self._mock or self._ser is None:
                log.info("[MOCK UART] %s", cmd)
            else:
                try:
                    self._ser.write(line.encode("ascii"))
                    self._ser.flush()
                except Exception as e:
                    log.error("UART write failed: %s — falling back to MOCK", e)
                    self._mock = True
            if self._on_send:
                try:
                    self._on_send(cmd)
                except Exception:
                    log.exception("on_send callback failed")
        return cmd

    def read_line(self) -> str:
        """Read one line of response from the ESP32 (status messages, etc.)."""
        if self._mock or self._ser is None:
            return ""
        try:
            return self._ser.readline().decode("ascii", errors="ignore").strip()
        except Exception as e:
            log.debug("UART read failed: %s", e)
            return ""

    # ---------- high-level commands (match walk_test.ino) ----------

    def stand(self) -> None:
        self.send("stand")

    def walk(self) -> None:
        self.send("walk")

    def stop(self) -> None:
        self.send("stop")

    def info(self) -> None:
        self.send("info")

    def speed(self, level: int) -> None:
        if level not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}, got {level}")
        self.send(str(level))

    def all_servos(self, angle: int) -> None:
        """Send a bare number — walk_test.ino routes this to every servo."""
        angle = int(angle)
        self.send(str(angle))

    def leg(self, leg: str, angle: int) -> None:
        """`lf 45` etc. — sets all 3 joints of a single leg to the same angle."""
        leg = leg.lower()
        if leg not in LEGS:
            raise ValueError(f"leg must be one of {LEGS}, got {leg}")
        self.send(f"{leg} {int(angle)}")

    def joint(self, leg: str, joint: str, angle: int) -> None:
        """`lfa 45`, `rbg 80`, ... — sets one servo only."""
        leg = leg.lower()
        joint = joint.lower()
        if leg not in LEGS:
            raise ValueError(f"leg must be one of {LEGS}, got {leg}")
        if joint not in JOINTS:
            raise ValueError(f"joint must be one of {JOINTS}, got {joint}")
        self.send(f"{leg}{joint} {int(angle)}")

    # ---------- convenience wrappers per leg ----------

    def lf(self, angle: int) -> None: self.leg("lf", angle)
    def lb(self, angle: int) -> None: self.leg("lb", angle)
    def rf(self, angle: int) -> None: self.leg("rf", angle)
    def rb(self, angle: int) -> None: self.leg("rb", angle)

    def lfa(self, a: int) -> None: self.joint("lf", "a", a)
    def lfb(self, a: int) -> None: self.joint("lf", "b", a)
    def lfg(self, a: int) -> None: self.joint("lf", "g", a)
    def lba(self, a: int) -> None: self.joint("lb", "a", a)
    def lbb(self, a: int) -> None: self.joint("lb", "b", a)
    def lbg(self, a: int) -> None: self.joint("lb", "g", a)
    def rfa(self, a: int) -> None: self.joint("rf", "a", a)
    def rfb(self, a: int) -> None: self.joint("rf", "b", a)
    def rfg(self, a: int) -> None: self.joint("rf", "g", a)
    def rba(self, a: int) -> None: self.joint("rb", "a", a)
    def rbb(self, a: int) -> None: self.joint("rb", "b", a)
    def rbg(self, a: int) -> None: self.joint("rb", "g", a)

    # ---------- composite poses / fun moves ----------
    # The REVO UART firmware handles the interpolation server-side, so each
    # of these is just one line on the wire.

    def sit(self) -> None:       self.send("sit")
    def tail_wag(self) -> None:  self.send("tail_wag")
    def bark(self) -> None:      self.send("bark")
    def greet(self) -> None:     self.send("greet")
    def lean_left(self) -> None: self.send("left")
    def lean_right(self) -> None: self.send("right")
    def backward(self) -> None:  self.send("backward")
    def ping(self) -> str:
        self.send("ping")
        return self.read_line()


_singleton: Optional[UartBridge] = None


def get_bridge(settings=None, on_send: Optional[Callable[[str], None]] = None) -> UartBridge:
    global _singleton
    if _singleton is None:
        if settings is None:
            from .config import settings as _s
            settings = _s
        _singleton = UartBridge(
            device=settings.serial_device,
            baud=settings.serial_baud,
            timeout=settings.serial_timeout_s,
            force_mock=settings.force_mock_uart,
            on_send=on_send,
        )
    elif on_send is not None:
        _singleton._on_send = on_send
    return _singleton
