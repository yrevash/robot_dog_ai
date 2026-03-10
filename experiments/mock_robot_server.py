#!/usr/bin/env python3
"""
experiments/mock_robot_server.py
=================================
Phase 6 helper — simple HTTP server to receive and timestamp robot commands.

Listens on configurable host:port (default localhost:8080).
Endpoint: POST /cmd — receives JSON {person, command, source, timestamp}

For each received command:
  - Logs with a server-side receive timestamp
  - Computes round-trip latency: server_time - payload["timestamp"]
  - Appends to results/phase6/received_commands.csv
  - Prints running stats: total received, commands/min, avg latency

Runs until Ctrl+C (SIGINT) or SIGTERM.

Usage:
  python experiments/mock_robot_server.py
  python experiments/mock_robot_server.py --host 0.0.0.0 --port 8080

  # Then in another terminal:
  python experiments/latency_measure.py --server-url http://localhost:8080/cmd
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import setup_logging, get_results_dir  # noqa: E402

# ── Module-level logger (set up in main(), used by handler) ──────────────────
log: logging.Logger = logging.getLogger("phase6_mock_server")

# ── Shared mutable state (protected by _lock) ─────────────────────────────────
_lock               = threading.Lock()
_received_commands: List[dict] = []
_seq_no             = 0
_total_latency_ms   = 0.0
_start_time: float  = 0.0
_csv_path: Optional[Path] = None

# CSV columns
_CSV_FIELDNAMES = [
    "seq_no", "server_timestamp", "payload_timestamp",
    "latency_ms", "person", "command", "source",
]


# ══════════════════════════════════════════════════════════════════════════════
# CSV writer (append-safe, opens per-write to avoid buffering issues)
# ══════════════════════════════════════════════════════════════════════════════

def _append_csv_row(row: dict) -> None:
    if _csv_path is None:
        return
    write_header = not _csv_path.exists() or _csv_path.stat().st_size == 0
    with open(_csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# Stats printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_stats(seq: int, avg_latency: float) -> None:
    elapsed_s   = time.time() - _start_time
    elapsed_min = max(elapsed_s / 60.0, 1e-6)
    cmd_per_min = seq / elapsed_min
    log.info(
        "[STATS] total=%d  rate=%.1f cmd/min  avg_latency=%.2f ms  uptime=%.1f s",
        seq, cmd_per_min, avg_latency, elapsed_s,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTTP request handler
# ══════════════════════════════════════════════════════════════════════════════

class RobotCommandHandler(BaseHTTPRequestHandler):
    """
    Handles POST /cmd requests containing JSON robot command payloads.

    Expected JSON body:
      {
        "person":    "<string>",
        "command":   "<string>",
        "source":    "<string>",
        "timestamp": <float UNIX seconds>
      }

    Response:
      200 OK  — {"status": "ok", "seq_no": N, "server_timestamp": T}
      400     — on malformed JSON or missing fields
      404     — on unknown path
      405     — on non-POST method to /cmd
    """

    # Silence default access log to keep output readable
    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        pass

    def log_error(self, fmt: str, *args) -> None:  # type: ignore[override]
        log.debug("HTTP error: " + fmt, *args)

    # ── Response helpers ──────────────────────────────────────────────────────

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, code: int, message: str) -> None:
        self._send_json(code, {"status": "error", "message": message})

    # ── Routing ───────────────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: N802 (HTTP method name convention)
        if self.path not in ("/cmd", "/cmd/"):
            self._send_error_json(404, f"Unknown path: {self.path}")
            return
        self._handle_cmd()

    def do_GET(self) -> None:  # noqa: N802
        """Health-check endpoint — useful for confirming server is up."""
        if self.path in ("/health", "/health/"):
            self._send_json(200, {"status": "ok", "uptime_s": time.time() - _start_time})
        else:
            self._send_error_json(404, f"Unknown path: {self.path}")

    # ── Command handler ───────────────────────────────────────────────────────

    def _handle_cmd(self) -> None:
        global _seq_no, _total_latency_ms

        server_ts = time.time()

        # Read body
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length) if length > 0 else b""

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Malformed JSON body: %s", exc)
            self._send_error_json(400, f"Invalid JSON: {exc}")
            return

        # Validate expected fields
        required = {"person", "command", "source", "timestamp"}
        missing  = required - payload.keys()
        if missing:
            log.warning("Missing fields in payload: %s", missing)
            self._send_error_json(400, f"Missing fields: {missing}")
            return

        try:
            payload_ts  = float(payload["timestamp"])
        except (TypeError, ValueError):
            payload_ts  = server_ts

        latency_ms = (server_ts - payload_ts) * 1000.0
        # Guard against clock skew / negative latency
        latency_ms = max(0.0, latency_ms)

        with _lock:
            _seq_no           += 1
            seq                = _seq_no
            _total_latency_ms += latency_ms
            avg_latency        = _total_latency_ms / _seq_no

            row = {
                "seq_no":            seq,
                "server_timestamp":  f"{server_ts:.6f}",
                "payload_timestamp": f"{payload_ts:.6f}",
                "latency_ms":        f"{latency_ms:.3f}",
                "person":            str(payload.get("person",  "")),
                "command":           str(payload.get("command", "")),
                "source":            str(payload.get("source",  "")),
            }
            _received_commands.append(row)

        # Log the command
        log.info(
            "[CMD #%04d] person=%-12s command=%-12s source=%-20s latency=%.2f ms",
            seq,
            row["person"],
            row["command"],
            row["source"],
            latency_ms,
        )

        # Persist to CSV
        _append_csv_row(row)

        # Print stats every 10 commands
        if seq % 10 == 0:
            _print_stats(seq, avg_latency)

        # Respond
        self._send_json(200, {
            "status":           "ok",
            "seq_no":           seq,
            "server_timestamp": server_ts,
            "latency_ms":       round(latency_ms, 3),
        })


# ══════════════════════════════════════════════════════════════════════════════
# Threading HTTP server
# ══════════════════════════════════════════════════════════════════════════════

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """
    HTTPServer that handles each request in a new thread.
    ThreadingMixIn sets daemon_threads=False by default, so threads finish
    before the process exits.
    """
    daemon_threads = True   # Allow clean Ctrl+C shutdown
    allow_reuse_address = True


# ══════════════════════════════════════════════════════════════════════════════
# Shutdown helpers
# ══════════════════════════════════════════════════════════════════════════════

_server_ref: Optional[ThreadingHTTPServer] = None


def _shutdown_handler(signum: int, frame) -> None:  # type: ignore[type-arg]
    """SIGINT / SIGTERM handler — shuts down the server gracefully."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    log.info("Received %s — shutting down …", sig_name)
    if _server_ref is not None:
        # shutdown() blocks until serve_forever() returns
        t = threading.Thread(target=_server_ref.shutdown, daemon=True)
        t.start()


def _print_final_summary() -> None:
    with _lock:
        seq        = _seq_no
        total_lat  = _total_latency_ms

    elapsed_s   = time.time() - _start_time
    avg_latency = total_lat / seq if seq > 0 else 0.0
    cmd_per_min = seq / max(elapsed_s / 60.0, 1e-6)

    log.info("=" * 60)
    log.info("Final summary")
    log.info("  Total commands received : %d", seq)
    log.info("  Avg round-trip latency  : %.2f ms", avg_latency)
    log.info("  Commands / minute       : %.1f", cmd_per_min)
    log.info("  Uptime                  : %.1f s", elapsed_s)
    if _csv_path is not None:
        log.info("  CSV saved to            : %s", _csv_path)
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 6 — Mock robot HTTP server for latency measurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",  type=str, default="localhost",
                   help="Interface to listen on (use 0.0.0.0 for all interfaces).")
    p.add_argument("--port",  type=int, default=8080,
                   help="TCP port to listen on.")
    p.add_argument("--no-csv", action="store_true",
                   help="Disable CSV logging (commands still printed to stdout).")
    return p.parse_args()


def main() -> None:
    global log, _csv_path, _start_time, _server_ref

    args = _parse_args()
    log  = setup_logging("phase6_mock_server")

    out_dir = get_results_dir("phase6")
    if not args.no_csv:
        _csv_path = out_dir / "received_commands.csv"
        # Truncate / create fresh CSV for this run
        if _csv_path.exists():
            _csv_path.unlink()
        log.info("CSV output: %s", _csv_path)

    # ── Register signal handlers ───────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # ── Start server ──────────────────────────────────────────────────────────
    server = ThreadingHTTPServer((args.host, args.port), RobotCommandHandler)
    _server_ref = server
    _start_time  = time.time()

    log.info("=" * 60)
    log.info("Mock Robot Server started")
    log.info("  Listening on : http://%s:%d", args.host, args.port)
    log.info("  Command URL  : http://%s:%d/cmd", args.host, args.port)
    log.info("  Health URL   : http://%s:%d/health", args.host, args.port)
    log.info("  Press Ctrl+C to stop")
    log.info("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Caught before signal handler fires on some platforms
        log.info("KeyboardInterrupt — shutting down …")
    finally:
        server.server_close()
        _print_final_summary()


if __name__ == "__main__":
    main()
