#!/usr/bin/env bash
# Start the REVO Pi deploy server.
# Run from repo root: ./pi_deploy/run.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Pick the right serial device automatically:
#   Raspberry Pi OS → /dev/serial0 (symlink, always points to the active UART)
#   Ubuntu on Pi 5  → /dev/ttyAMA0 (no symlink — use the real device name)
#   Neither exists  → mock mode so the app still boots for UI development
if [ -z "${REVO_SERIAL:-}" ]; then
  if   [ -e "/dev/serial0" ]; then export REVO_SERIAL=/dev/serial0
  elif [ -e "/dev/ttyAMA0" ]; then export REVO_SERIAL=/dev/ttyAMA0
  else
    export REVO_MOCK_UART=1
    echo "[run.sh] no UART device found — running with REVO_MOCK_UART=1"
  fi
fi
echo "[run.sh] REVO_SERIAL=${REVO_SERIAL:-<unset>}  REVO_MOCK_UART=${REVO_MOCK_UART:-0}"

exec uvicorn pi_deploy.app.main:app \
  --host 0.0.0.0 \
  --port "${REVO_PORT:-8080}" \
  --log-level info
