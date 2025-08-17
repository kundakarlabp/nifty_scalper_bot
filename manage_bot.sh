#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo "[manage_bot] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

show_env() {
  log "Env summary:"
  printf '  %-22s %s\n' "PYTHONPATH"            "${PYTHONPATH:-/app}"
  printf '  %-22s %s\n' "PORT"                  "${PORT:-8000}"
  printf '  %-22s %s\n' "ENABLE_LIVE_TRADING"   "${ENABLE_LIVE_TRADING:-<unset>}"
  printf '  %-22s %s\n' "ENABLE_TELEGRAM"       "${ENABLE_TELEGRAM:-<unset>}"
  printf '  %-22s %s\n' "TELEGRAM_BOT_TOKEN"    "$([ -n "${TELEGRAM_BOT_TOKEN:-}" ] && echo '***' || echo '(empty)')"
  printf '  %-22s %s\n' "TELEGRAM_CHAT_ID"      "${TELEGRAM_CHAT_ID:-'(not set)'}"
  printf '  %-22s %s\n' "ZERODHA_API_KEY"       "$([ -n "${ZERODHA_API_KEY:-}" ] && echo '***' || echo '(empty)')"
  printf '  %-22s %s\n' "KITE_ACCESS_TOKEN"     "$([ -n "${KITE_ACCESS_TOKEN:-${ZERODHA_ACCESS_TOKEN:-}}" ] && echo '***' || echo '(empty)')"
}

cleanup() {
  log "Caught termination signal. Stopping trader…"
  if command -v python >/dev/null 2>&1; then
    python -m src.main stop || true
  fi
}
trap cleanup SIGTERM SIGINT

CMD="${1:-run}"
case "$CMD" in
  run)
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${PYTHONPATH:-/app}"
    export PORT="${PORT:-8000}"
    show_env
    log "Starting trader (supervised)."
    # Use exec so Python receives signals directly
    exec python -m src.main run
    ;;
  start)
    exec python -m src.main start
    ;;
  stop)
    exec python -m src.main stop
    ;;
  status)
    exec python -m src.main status
    ;;
  *)
    echo "Usage: $0 {run|start|stop|status}" >&2
    exit 64
    ;;
esac