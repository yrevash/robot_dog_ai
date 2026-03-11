"""
power_state.py
==============
Framework-agnostic power management state machine for REVO.

States:
    ACTIVE      — full inference, camera running, gesture detection on
    POWER_SAVE  — reduced frame rate, gesture detection off, face detection only
    POWER_OFF   — camera released, all inference stopped, waiting for wake trigger

Transitions:
    ACTIVE → POWER_SAVE   after `idle_to_save_sec` seconds with no activity
    POWER_SAVE → POWER_OFF  after `save_to_off_sec` total idle seconds
    Any → ACTIVE           on wake() call or face detection in POWER_SAVE
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable, Optional


class PowerState(Enum):
    ACTIVE = auto()
    POWER_SAVE = auto()
    POWER_OFF = auto()


class PowerManager:
    """Tracks idle time and drives state transitions via callbacks."""

    def __init__(
        self,
        idle_to_save_sec: float = 900.0,   # 15 minutes
        save_to_off_sec: float = 1800.0,   # 30 minutes total
        on_enter_active: Optional[Callable[[], None]] = None,
        on_enter_power_save: Optional[Callable[[], None]] = None,
        on_enter_power_off: Optional[Callable[[], None]] = None,
    ) -> None:
        self.idle_to_save_sec = idle_to_save_sec
        self.save_to_off_sec = save_to_off_sec
        self._on_enter_active = on_enter_active
        self._on_enter_power_save = on_enter_power_save
        self._on_enter_power_off = on_enter_power_off

        self._state = PowerState.ACTIVE
        self._last_activity = time.monotonic()

    @property
    def state(self) -> PowerState:
        return self._state

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def report_activity(self) -> None:
        """Call when a face is recognized or a gesture command fires."""
        self._last_activity = time.monotonic()
        if self._state == PowerState.POWER_SAVE:
            self._transition(PowerState.ACTIVE)

    def wake(self) -> None:
        """Explicit wake trigger (button press). Works from any state."""
        self._last_activity = time.monotonic()
        if self._state != PowerState.ACTIVE:
            self._transition(PowerState.ACTIVE)

    def tick(self) -> None:
        """Call every update loop iteration to check for timeout transitions."""
        idle = self.idle_seconds

        if self._state == PowerState.ACTIVE:
            if idle >= self.idle_to_save_sec:
                self._transition(PowerState.POWER_SAVE)
        elif self._state == PowerState.POWER_SAVE:
            if idle >= self.save_to_off_sec:
                self._transition(PowerState.POWER_OFF)

    def _transition(self, new_state: PowerState) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        cb = {
            PowerState.ACTIVE: self._on_enter_active,
            PowerState.POWER_SAVE: self._on_enter_power_save,
            PowerState.POWER_OFF: self._on_enter_power_off,
        }.get(new_state)
        if cb is not None:
            cb()
