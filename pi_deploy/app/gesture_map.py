"""
Translate gesture names used by the face pipeline into a single line on the
wire for the REVO UART firmware (`tools/revo_uart/revo_uart.ino`).

The firmware understands high-level commands natively (sit, tail_wag, bark,
greet, left, right, backward), so each gesture becomes exactly one line —
the ESP32 handles the 30-step interpolation for composite poses.

Gestures (see docs/GESTURE_CHEAT_SHEET.md):
    forward, backward, left, right, bark, stand, tail_wag, walk, sit, stop, greet
"""
from __future__ import annotations

from typing import Any, Callable, Dict
from .uart_bridge import UartBridge


def _forward(b: UartBridge):  b.walk()
def _stop(b: UartBridge):     b.stop()
def _stand(b: UartBridge):    b.stand()
def _sit(b: UartBridge):      b.send("sit")
def _bark(b: UartBridge):     b.send("bark")
def _tail_wag(b: UartBridge): b.send("tail_wag")
def _left(b: UartBridge):     b.send("left")
def _right(b: UartBridge):    b.send("right")
def _backward(b: UartBridge): b.send("backward")
def _greet(b: UartBridge):    b.send("greet")


GESTURE_TO_ESP: Dict[str, Callable[[UartBridge], None]] = {
    "walk": _forward,
    "forward": _forward,
    "stop": _stop,
    "stand": _stand,
    "sit": _sit,
    "bark": _bark,
    "tail_wag": _tail_wag,
    "left": _left,
    "right": _right,
    "backward": _backward,
    "greet": _greet,
}


_WALKING_TRUE = {"walk", "forward"}
_WALKING_FALSE = {"stop", "stand", "sit", "backward"}


def walking_effect(gesture: str) -> Dict[str, Any]:
    """Return the RobotState delta a gesture implies for the `walking` flag."""
    if gesture in _WALKING_TRUE:
        return {"walking": True}
    if gesture in _WALKING_FALSE:
        return {"walking": False}
    return {}


def dispatch(gesture: str, bridge: UartBridge) -> bool:
    """Run the ESP32 sequence for a gesture. Returns True if recognised."""
    fn = GESTURE_TO_ESP.get(gesture)
    if fn is None:
        return False
    fn(bridge)
    return True
