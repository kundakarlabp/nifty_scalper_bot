#!/usr/bin/env bash
# src/manage_bot.sh  (place at repo root if that's where your Procfile points)
set -euo pipefail

# --- logging helpers ---------------------------------------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] [manage_bot] $*"; }

# Print a tiny env summary without leaking secrets.
print_env_summary() {
  log "Env summary:"
  echo "PYTHONPATH     : ${PYTHONPATH:-<unset>}"
  echo "PORT           : ${PORT:-<unset>}"
  echo "ENABLE_LIVE_TRADING : ${ENABLE_LIVE_TRADING:-<unset>}"
  echo "ENABLE_TELEGRAM     : ${ENABLE_TELEGRAM:-<unset>}"
  # Redacted indicators (avoid printing secrets):
  [[ -n "${TELEGRAM_BOT_TOKEN:-}" ]] && echo "TELEGRAM_BOT_TOKEN : <set>" || echo "TELEGRAM_BOT_TOKEN : <unset>"
  [[ -n "${TELEGRAM_CHAT_ID:-}" ]] && echo "TELEGRAM_CHAT_ID   : <set>" || echo "TELEGRAM_CHAT_ID   : <unset>"
  [[ -n "${ZERODHA_API_KEY:-}" ]] && echo "ZERODHA_API_KEY    : <set>" || echo "ZERODHA_API_KEY    : <unset>"
  [[ -n "${KITE_ACCESS_TOKEN:-${ZERODHA_ACCESS_TOKEN:-}}" ]] && echo "KITE_ACCESS_TOKEN  : <set>" || echo "KITE_ACCESS_TOKEN  : <unset>"
}

# --- graceful shutdown -------------------------------------------------------
stopping=false
_term() {
  $stopping && return
  stopping=true
  log "Caught SIGTERM, asking app to stop..."
  # Best-effort stop hook (safe if your src.main exposes it; otherwise no-op)
  python -m src.main stop >/dev/null 2>&1 || true
  exit 0
}
trap _term SIGTERM SIGINT

# Ensure Python logs flush immediately (Railway log tailing)
export PYTHONUNBUFFERED=1
# If you keep your code inside /app in Railway, set a reasonable PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-/app}"

cmd="${1:-run}"

case "$cmd" in
  run)
    print_env_summary
    log "Starting trader (supervised)…"
    # Simple supervisor: restart on unexpected crash, exit cleanly on code 0.
    # Prevents Railway from flapping if your process exits due to transient issues.
    while true; do
      set +e
      python -m src.main run
      code=$?
      set -e
      if [[ $code -eq 0 ]]; then
        log "Trader exited cleanly (code 0). Not restarting."
        break
      fi
      $stopping && break
      log "Trader crashed with code $code. Restarting in 3s…"
      sleep 3
    done
    ;;

  start)
    log "Sending start to running process (best-effort)…"
    python -m src.main start
    ;;

  stop)
    log "Sending stop to running process (best-effort)…"
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