#!/usr/bin/env bash
set -euo pipefail

# Graceful shutdown
_term() {
  echo "[manage_bot] Caught SIGTERM, stopping trader..."
  python -m src.main stop || true
  exit 0
}
trap _term SIGTERM SIGINT

CMD="${1:-run}"

# Print environment summary (helps debugging on Railway)
echo "[manage_bot] $(date '+%F %T') Env summary:"
echo "  PYTHONPATH: ${PYTHONPATH:-}"
echo "  PORT:       ${PORT:-}"
echo "  ENABLE_LIVE_TRADING: ${ENABLE_LIVE_TRADING:-<unset>}"
echo "  ENABLE_TELEGRAM:     ${ENABLE_TELEGRAM:-<unset>}"
echo "  TELEGRAM_BOT_TOKEN:  ${TELEGRAM_BOT_TOKEN:-<empty>}"
echo "  TELEGRAM_CHAT_ID:    ${TELEGRAM_CHAT_ID:-'(not set)'}"
echo "  ZERODHA_API_KEY:     ${ZERODHA_API_KEY:-<empty>}"
echo "  KITE_ACCESS_TOKEN:   ${KITE_ACCESS_TOKEN:-<empty>}"

case "$CMD" in
  run)
    echo "[manage_bot] Starting trader (supervised)..."
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