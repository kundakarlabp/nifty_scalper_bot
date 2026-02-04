#!/usr/bin/env bash
set -euo pipefail

# Deployment logging for Railway
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Nifty Scalper Bot - Starting deployment..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Environment: Python path = $PYTHONPATH"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Current directory: $(pwd)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] User: $(whoami)"


_term() {
  kill -TERM "$child" 2>/dev/null || true
}

trap _term SIGTERM

export PYTHONPATH="${PYTHONPATH:-/app/src}"
python -m nifty_scalper_bot.app &
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting nifty_scalper_bot app process (PID: $!)..."

child=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for nifty_scalper_bot process to complete..."

wait "$child"
