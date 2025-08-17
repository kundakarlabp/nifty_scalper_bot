#!/usr/bin/env bash
# src/scripts/manage_bot.sh
# Robust process manager for Railway deployments.
# - Single-instance lock to avoid duplicate Telegram pollers
# - Clean SIGTERM -> tell bot to stop -> exit
# - Optional auto-restart with exponential backoff on crash
# - Small env summary (safely masked) for debugging on Railway

set -Eeuo pipefail

APP_MODULE="src.main"
LOCK_FILE="/tmp/nifty_scalper.lock"
LOG_PREFIX="[manage_bot]"
MAX_RESTARTS="${MAX_RESTARTS:-10}"          # how many crash restarts before giving up
BACKOFF_START="${BACKOFF_START:-2}"         # seconds
BACKOFF_MAX="${BACKOFF_MAX:-30}"            # seconds
RAILWAY_HEALTH_PATH="${RAILWAY_HEALTH_PATH:-/health}"

mask() {
  # mask a secret but keep last 4 chars when present
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo "(empty)"
  elif [[ "${#s}" -le 8 ]]; then
    echo "****"
  else
    echo "*****${s: -4}"
  fi
}

print_env_summary() {
  echo "$LOG_PREFIX Env summary:"
  echo "$LOG_PREFIX  PYTHONPATH: ${PYTHONPATH:-"(default)"}"
  echo "$LOG_PREFIX  PORT:       ${PORT:-"(not set)"}"
  echo "$LOG_PREFIX  TELEGRAM_BOT_TOKEN: $(mask "${TELEGRAM_BOT_TOKEN:-}")"
  echo "$LOG_PREFIX  TELEGRAM_CHAT_ID:   ${TELEGRAM_CHAT_ID:-"(not set)"}"
  echo "$LOG_PREFIX  ZERODHA_API_KEY:    $(mask "${ZERODHA_API_KEY:-}")"
  echo "$LOG_PREFIX  KITE_ACCESS_TOKEN:  $(mask "${KITE_ACCESS_TOKEN:-${ZERODHA_ACCESS_TOKEN:-}}")"
  echo "$LOG_PREFIX  ENABLE_TELEGRAM:    ${ENABLE_TELEGRAM:-"(not set)"}"
  echo "$LOG_PREFIX  ENABLE_LIVE_TRADING:${ENABLE_LIVE_TRADING:-"(not set)"}"
}

# --- locking (prevent multiple instances) ---------------------------------
exec 9>"$LOCK_FILE" || true
if ! flock -n 9; then
  echo "$LOG_PREFIX Another instance appears to be running. Exiting."
  exit 0
fi

# --- graceful shutdown -----------------------------------------------------
stop_bot() {
  echo "$LOG_PREFIX Sending graceful stop to bot…"
  # Best-effort; if your src.main supports it:
  python -m "$APP_MODULE" stop >/dev/null 2>&1 || true
}

_term() {
  echo "$LOG_PREFIX Caught SIGTERM/SIGINT."
  stop_bot
  # give python time to flush/exit
  sleep 1
  echo "$LOG_PREFIX Exiting."
  exit 0
}
trap _term SIGTERM SIGINT

# --- helpers ---------------------------------------------------------------
run_bot_once() {
  # This is the long-lived run command that starts the trader and its polling worker
  python -m "$APP_MODULE" run
}

health_ping() {
  # Railway exposes $PORT; our app binds to it. This is optional/no-op if curl missing.
  if command -v curl >/dev/null 2>&1 && [[ -n "${PORT:-}" ]]; then
    curl -fsS "http://127.0.0.1:${PORT}${RAILWAY_HEALTH_PATH}" >/dev/null 2>&1 || true
  fi
}

usage() {
  cat <<EOF
Usage: $0 {run|start|stop|status}

Commands:
  run     Start the bot (supervised; auto-restarts on crash).
  start   Ask running app to start trading (via src.main start).
  stop    Ask running app to stop trading (via src.main stop).
  status  Query current status (via src.main status).
Env (optional):
  MAX_RESTARTS (default: ${MAX_RESTARTS})
  BACKOFF_START (default: ${BACKOFF_START}s)
  BACKOFF_MAX (default: ${BACKOFF_MAX}s)
EOF
}

# --- command router --------------------------------------------------------
CMD="${1:-run}"

case "$CMD" in
  run)
    print_env_summary
    echo "$LOG_PREFIX Starting trader (supervised)..."
    restart=0
    backoff="$BACKOFF_START"
    while :; do
      set +e
      run_bot_once
      code=$?
      set -e

      # If exited cleanly, just stop loop
      if [[ "$code" -eq 0 ]]; then
        echo "$LOG_PREFIX Bot exited cleanly (code 0)."
        break
      fi

      restart=$((restart+1))
      if (( restart > MAX_RESTARTS )); then
        echo "$LOG_PREFIX Reached MAX_RESTARTS=$MAX_RESTARTS. Not restarting."
        break
      fi

      echo "$LOG_PREFIX Bot crashed (exit $code). Restart #$restart in ${backoff}s..."
      sleep "$backoff"
      # increase backoff up to BACKOFF_MAX
      if (( backoff < BACKOFF_MAX )); then
        backoff=$(( backoff * 2 ))
        if (( backoff > BACKOFF_MAX )); then
          backoff="$BACKOFF_MAX"
        fi
      fi

      health_ping
    done
    ;;

  start)
    echo "$LOG_PREFIX Requesting /start…"
    python -m "$APP_MODULE" start
    ;;

  stop)
    echo "$LOG_PREFIX Requesting /stop…"
    python -m "$APP_MODULE" stop
    ;;

  status)
    echo "$LOG_PREFIX Requesting /status…"
    python -m "$APP_MODULE" status
    ;;

  *)
    usage
    exit 1
    ;;
esac