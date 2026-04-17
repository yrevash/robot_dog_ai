from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class RobotState:
    mode: str = "pi_camera"                    # pi_camera | laptop_client | laptop_stream
    authorized: Optional[str] = None
    last_gesture: Optional[str] = None
    last_command: Optional[str] = None
    last_command_ts: float = 0.0
    speed: int = 3
    walking: bool = False
    uart_connected: bool = False
    ai_enabled: bool = False     # becomes True only after worker.start() succeeds
    fps: float = 0.0
    log: List[str] = field(default_factory=list)

    def snapshot(self) -> dict:
        return asdict(self)


class StateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = RobotState()
        self._subscribers: list = []

    def update(self, **fields) -> dict:
        """Update fields and broadcast only if any value actually changed."""
        changed = False
        with self._lock:
            for k, v in fields.items():
                if getattr(self._state, k, None) != v:
                    setattr(self._state, k, v)
                    changed = True
            snap = self._state.snapshot()
        if changed:
            self._notify(snap)
        return snap

    def log_line(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self._lock:
            self._state.log.append(line)
            self._state.log = self._state.log[-50:]
            snap = self._state.snapshot()
        self._notify(snap)

    def get(self) -> dict:
        with self._lock:
            return self._state.snapshot()

    def subscribe(self, fn) -> None:
        with self._lock:
            self._subscribers.append(fn)

    def unsubscribe(self, fn) -> None:
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def _notify(self, snap: dict) -> None:
        for fn in list(self._subscribers):
            try:
                fn(snap)
            except Exception:
                pass


store = StateStore()
