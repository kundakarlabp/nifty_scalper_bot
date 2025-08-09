#!/usr/bin/env bash
# manage_bot.sh – Control script for Nifty Scalper Bot

set -euo pipefail

# === Resolve repo root so the script works from anywhere ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

# === Configuration ===
PYTHON="python3"
[[ -x ".venv/bin/python" ]] && PYTHON=".venv/bin/python"

MODULE="src.main"
ARGS="start"

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/scalper_bot.log"
PID_FILE="bot.pid"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Simple 5 MB rotation
rotate_logs() {
  local max_bytes=$((5 * 1024 * 1024))
  if [[ -f "$LOG_FILE" ]]; then
    local size
    size=$(wc -c < "$LOG_FILE" || echo 0)
    if [[ "$size" -ge "$max_bytes" ]]; then
      mv -f "$LOG_FILE" "${LOG_FILE}.$(date +%Y%m%d_%H%M%S)"
      : > "$LOG_FILE"
      echo "🌀 Rotated log to ${LOG_FILE}.*"
    fi
  fi
}

# === Helper Functions ===
get_pid() {
  [[ -f "$PID_FILE" ]] && cat "$PID_FILE" || true
}

is_running() {
  local pid
  pid="$(get_pid || true)"
  [[ -z "${pid:-}" ]] && return 1
  if kill -0 "$pid" 2>/dev/null; then
    # Optional sanity check: ensure the PID looks like our process
    if ps -p "$pid" -o cmd= | grep -q "$MODULE"; then
      return 0
    fi
  fi
  # Stale PID
  return 1
}

save_pid() {
  echo "$1" > "$PID_FILE"
}

clear_pid() {
  rm -f "$PID_FILE"
}

start_bot() {
  echo "🚀 Starting Nifty Scalper Bot..."
  if is_running; then
    echo "❌ Already running (PID: $(get_pid)). Use '$0 restart' to restart."
    exit 1
  fi

  rotate_logs

  # Run as a module to keep import paths correct
  nohup "$PYTHON" -m "$MODULE" "$ARGS" >> "$LOG_FILE" 2>&1 &
  save_pid $!

  sleep 1
  if is_running; then
    echo "✅ Bot started (PID: $(get_pid))"
    echo "📜 Logs: $LOG_FILE"
    echo "💡 To monitor: $0 monitor"
  else
    echo "❌ Failed to start. See logs: $LOG_FILE"
    exit 1
  fi
}

stop_bot() {
  echo "🛑 Stopping Nifty Scalper Bot..."
  if is_running; then
    local pid
    pid="$(get_pid)"
    kill "$pid" 2>/dev/null || true

    # Grace period up to 10s
    for _ in {1..10}; do
      if ! is_running; then break; fi
      sleep 1
    done

    if is_running; then
      echo "⚠️ Forcing shutdown..."
      kill -9 "$pid" 2>/dev/null || true
      sleep 1
    fi

    clear_pid
    echo "✅ Stopped."
  else
    echo "💤 Not running."
    clear_pid || true
  fi
}

case "${1:-status}" in
  start)
    start_bot
    ;;
  stop)
    stop_bot
    ;;
  restart)
    echo "🔄 Restarting..."
    stop_bot
    sleep 2
    start_bot
    ;;
  status)
    echo "📊 Status:"
    if is_running; then
      echo "  ✅ RUNNING (PID: $(get_pid))"
    else
      echo "  ❌ STOPPED"
      if [[ -f "$PID_FILE" ]]; then
        echo "  ⚠️ Stale PID: $(cat "$PID_FILE")"
      else
        echo "  ℹ️ No PID file."
      fi
    fi
    ;;
  logs)
    echo "📄 Last 40 log lines:"
    [[ -f "$LOG_FILE" ]] && tail -n 40 "$LOG_FILE" || echo "⚠️ Log file missing."
    ;;
  monitor)
    echo "🔍 Live Logs (Ctrl+C to exit)"
    [[ ! -f "$LOG_FILE" ]] && : > "$LOG_FILE"
    # trap to cleanly stop tail
    trap 'exit 0' INT TERM
    tail -F "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|monitor}"
    echo "Examples:"
    echo "  $0 start     # Start bot"
    echo "  $0 stop      # Stop bot"
    echo "  $0 status    # Check if running"
    echo "  $0 logs      # Show last 40 lines"
    echo "  $0 monitor   # Follow logs live"
    exit 1
    ;;
esac
