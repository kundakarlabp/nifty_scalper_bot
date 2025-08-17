# manage_bot.sh
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="nifty-scalper-bot"
LOCK_FILE="/tmp/${APP_NAME}.pid"

# --- helpers ---------------------------------------------------------------

is_running() {
  [[ -f "$LOCK_FILE" ]] || return 1
  local pid
  pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  if ps -p "$pid" > /dev/null 2>&1; then
    return 0
  else
    # stale lock
    rm -f "$LOCK_FILE"
    return 1
  fi
}

start_cmd() {
  # Optional: source a local .env for local dev; Railway doesn’t need this.
  if [[ -f ".env" ]]; then
    set -a
    # shellcheck source=/dev/null
    . ./.env
    set +a
  fi

  # Guard against double starts
  if is_running; then
    echo "[manage_bot] Already running (pid $(cat "$LOCK_FILE"))."
    exit 0
  fi

  export PYTHONUNBUFFERED=1
  export APP_ENV="${APP_ENV:-production}"

  # Ensure Telegram polling has no competing listener process
  # (you already removed the old listener; this is just a reminder comment)

  # Start the python supervisor in the foreground
  echo "[manage_bot] Starting trader…"
  python -m src.main run &
  echo $! > "$LOCK_FILE"

  # Trap for Railway stop/redeploy
  trap '_stop_gracefully' TERM INT

  wait
}

_stop_gracefully() {
  echo "[manage_bot] Caught stop signal — stopping trader…"
  if is_running; then
    local pid
    pid="$(cat "$LOCK_FILE")"
    # Best-effort app-level stop (lets the bot flatten/cleanup)
    python -m src.main stop || true
    # Also send SIGTERM to the worker if it’s still alive
    if ps -p "$pid" > /dev/null 2>&1; then
      kill -TERM "$pid" 2>/dev/null || true
      # wait up to 15s
      for _ in {1..15}; do
        ps -p "$pid" > /dev/null 2>&1 || break
        sleep 1
      done
      # force if stubborn
      ps -p "$pid" > /dev/null 2>&1 && kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$LOCK_FILE"
  fi
  echo "[manage_bot] Stopped."
  exit 0
}

do_start() {
  start_cmd
}

do_stop() {
  if ! is_running; then
    echo "[manage_bot] Not running."
    exit 0
  fi
  local pid
  pid="$(cat "$LOCK_FILE")"
  echo "[manage_bot] Stopping pid ${pid}…"
  python -m src.main stop || true
  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..15}; do
    ps -p "$pid" > /dev/null 2>&1 || break
    sleep 1
  done
  ps -p "$pid" > /dev/null 2>&1 && kill -KILL "$pid" 2>/dev/null || true
  rm -f "$LOCK_FILE"
  echo "[manage_bot] Stopped."
}

do_status() {
  if is_running; then
    echo "[manage_bot] RUNNING (pid $(cat "$LOCK_FILE"))."
    python -m src.main status || true
  else
    echo "[manage_bot] NOT RUNNING."
  fi
}

do_restart() {
  do_stop || true
  do_start
}

# Convenience: quick live/shadow toggles at launch (optional)
do_live() {
  export ENABLE_LIVE_TRADING=true
  echo "[manage_bot] Live mode requested."
  do_start
}
do_shadow() {
  export ENABLE_LIVE_TRADING=false
  echo "[manage_bot] Shadow mode requested."
  do_start
}

# --- entrypoint ------------------------------------------------------------

cmd="${1:-run}"
case "$cmd" in
  run|start)     do_start ;;
  stop)          do_stop ;;
  status)        do_status ;;
  restart)       do_restart ;;
  live)          do_live ;;
  shadow)        do_shadow ;;
  *)
    cat <<USAGE
Usage: $0 {run|start|stop|status|restart|live|shadow}

run/start  - start the bot (default; single-instance guarded)
stop       - graceful stop
status     - show running status
restart    - stop then start
live       - start with ENABLE_LIVE_TRADING=true
shadow     - start with ENABLE_LIVE_TRADING=false
USAGE
    exit 1
    ;;
esac