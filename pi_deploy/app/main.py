"""
FastAPI entry point. Run with:

    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

Routes:
    GET  /                    — serves the web UI
    GET  /video_feed          — MJPEG stream of the latest annotated frame
    GET  /api/state           — current RobotState snapshot
    POST /api/mode            — switch camera mode (pi_camera | laptop_client | laptop_stream)
    POST /api/command         — raw ESP32 command (e.g. {"cmd": "walk"})
    POST /api/gesture         — high-level gesture name (for laptop_client mode)
    POST /api/speed/{level}   — 1..5
    POST /api/stand|walk|stop — shortcuts
    WS   /ws/frames           — laptop/browser pushes JPEG frames for laptop_stream mode
    WS   /ws/status           — broadcast RobotState updates to every connected UI
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ai_worker import get_worker
from .config import settings
from .gesture_map import GESTURE_TO_ESP, dispatch, walking_effect
from .robot_state import store
from .uart_bridge import get_bridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("revo.main")


class CommandBody(BaseModel):
    cmd: str


class GestureBody(BaseModel):
    gesture: str
    person: Optional[str] = None


class ModeBody(BaseModel):
    mode: str  # pi_camera | laptop_client | laptop_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = get_worker()
    if os.getenv("REVO_NO_AI", "0") == "1":
        log.warning("REVO_NO_AI=1 — skipping AI worker start (no camera, no inference)")
    else:
        try:
            worker.start()
        except Exception as e:
            log.error("AI worker failed to start: %s", e)
    yield
    worker.stop()


app = FastAPI(title="REVO Pi Deploy", lifespan=lifespan)


# ---------- static web UI ----------

app.mount("/static", StaticFiles(directory=str(settings.web_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(settings.web_dir / "index.html"))


# ---------- MJPEG stream ----------

_PLACEHOLDER_JPEG: Optional[bytes] = None


def _placeholder_jpeg() -> bytes:
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is None:
        ph = np.zeros((120, 160, 3), dtype="uint8")
        _, buf = cv2.imencode(".jpg", ph)
        _PLACEHOLDER_JPEG = buf.tobytes()
    return _PLACEHOLDER_JPEG


def _mjpeg_generator():
    boundary = b"--frame"
    interval = 1.0 / max(1, settings.target_fps)
    while True:
        jpeg = get_worker().latest_jpeg() or _placeholder_jpeg()
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ---------- state + commands ----------

@app.get("/api/state")
def api_state():
    return store.get()


@app.get("/api/gestures")
def api_gestures():
    return {"gestures": sorted(GESTURE_TO_ESP.keys())}


@app.post("/api/mode")
def api_mode(body: ModeBody):
    worker = get_worker()
    try:
        worker.set_mode(body.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return store.get()


@app.post("/api/command")
def api_command(body: CommandBody):
    bridge = get_bridge()
    cmd = bridge.send(body.cmd)
    store.update(last_command=cmd, last_command_ts=time.time())
    store.log_line(f"manual → {cmd}")
    return {"sent": cmd, "mock": bridge.is_mock}


@app.post("/api/gesture")
def api_gesture(body: GestureBody):
    bridge = get_bridge()
    if not dispatch(body.gesture, bridge):
        raise HTTPException(status_code=400, detail=f"unknown gesture: {body.gesture}")
    store.update(
        last_gesture=body.gesture,
        last_command=bridge.last_command,
        last_command_ts=time.time(),
        authorized=body.person or store.get().get("authorized"),
        **walking_effect(body.gesture),
    )
    store.log_line(f"{body.person or 'remote'} → {body.gesture} → {bridge.last_command}")
    return {"gesture": body.gesture, "sent": bridge.last_command, "mock": bridge.is_mock}


@app.post("/api/speed/{level}")
def api_speed(level: int):
    if level not in (1, 2, 3, 4, 5):
        raise HTTPException(status_code=400, detail="speed must be 1..5")
    bridge = get_bridge()
    bridge.speed(level)
    store.update(speed=level, last_command=str(level))
    store.log_line(f"speed → {level}")
    return {"speed": level}


@app.post("/api/ai/start")
def api_ai_start():
    worker = get_worker()
    try:
        worker.start()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    store.log_line("AI started")
    return {"ai_enabled": True}


@app.post("/api/ai/stop")
def api_ai_stop():
    get_worker().stop()
    store.log_line("AI stopped")
    return {"ai_enabled": False}


@app.post("/api/stand")
def api_stand():
    get_bridge().stand()
    store.update(walking=False, last_command="stand")
    store.log_line("stand")
    return {"ok": True}


@app.post("/api/walk")
def api_walk():
    get_bridge().walk()
    store.update(walking=True, last_command="walk")
    store.log_line("walk")
    return {"ok": True}


@app.post("/api/stop")
def api_stop():
    get_bridge().stop()
    store.update(walking=False, last_command="stop")
    store.log_line("stop")
    return {"ok": True}


# ---------- frame push (laptop_stream mode) ----------

@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    await ws.accept()
    worker = get_worker()
    try:
        while True:
            data = await ws.receive_bytes()
            worker.push_remote_jpeg(data)
    except WebSocketDisconnect:
        pass


# ---------- status broadcast ----------

@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    def on_update(snap: dict):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, snap)
        except Exception:
            pass

    store.subscribe(on_update)
    try:
        await ws.send_json(store.get())
        while True:
            snap = await queue.get()
            await ws.send_json(snap)
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe(on_update)
