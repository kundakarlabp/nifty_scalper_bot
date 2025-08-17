#!/usr/bin/env bash
set -euo pipefail

# ------------------- helpers -------------------
# Load .env (ignore comments/blank, strip CRLF). Does NOT echo values.
load_env() {
  local envfile=".env"
  [[ -f "$envfile" ]] || return 0
  # Only accept KEY=VALUE lines that look like environment vars
  # (prevents ".env: line 1: 9#: command not found" type issues).
  while IFS= read -r line; do
    line="${line//$'\r'/}"                 # strip CRLF if present
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      export "$line"
    fi
  done < "$envfile"
}

log() { echo "[manage_bot] $*"; }

graceful_stop() {
  log "Caught SIGTERM/SIGINT, stopping trader..."
  python -m src.main stop || true
  exit 0
}
trap graceful_stop SIGTERM SIGINT

# ------------------- main -------------------
CMD="${1:-run}"

# Load env quietly (so secrets in your .env never get printed)
load_env

# Optional informational log (no secrets)
if [[ -n "${POLL_SEC:-}" ]]; then
  log "Polling cadence override via .env POLL_SEC=${POLL_SEC}s"
fi

case "$CMD" in
  run)
    log "Starting trader (supervised)…"
    python -m src.main run
    ;;
  start)
    python -m src.main start
    ;;
  stop)
    python -m src.main stop
    ;;
  status)
    python -m src.main status
    ;;
  *)
    echo "Usage: $0 {run|start|stop|status}"
    exit 1
    ;;
esac