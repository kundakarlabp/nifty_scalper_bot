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
case "$CMD" in
  run)
    echo "[manage_bot] Starting trader (shadow mode by default)"
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