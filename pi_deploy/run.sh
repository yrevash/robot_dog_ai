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

# If /dev/serial0 is missing, force mock mode so the app still boots
if [ ! -e "/dev/serial0" ] && [ -z "${REVO_SERIAL:-}" ]; then
  export REVO_MOCK_UART=1
  echo "[run.sh] /dev/serial0 not found — running with REVO_MOCK_UART=1"
fi

exec uvicorn pi_deploy.app.main:app \
  --host 0.0.0.0 \
  --port "${REVO_PORT:-8080}" \
  --log-level info
