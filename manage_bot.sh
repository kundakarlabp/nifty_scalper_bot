#!/bin/bash
# manage_bot.sh – Control script for Nifty Scalper Bot

# === Configuration ===
SCRIPT="python3 -m src.main start"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/scalper_bot.log"
PID_FILE="bot.pid"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# === Helper Functions ===
get_pid() {
    [[ -f "$PID_FILE" ]] && cat "$PID_FILE"
}

is_running() {
    local pid=$(get_pid)
    [[ -n "$pid" ]] && kill -0 "$pid" > /dev/null 2>&1
}

save_pid() {
    echo "$1" > "$PID_FILE"
}

clear_pid() {
    rm -f "$PID_FILE"
}

# === Commands ===
case "${1:-status}" in
    start)
        echo "🚀 Starting Nifty Scalper Bot..."
        if is_running; then
            echo "❌ Already running (PID: $(get_pid)). Use '$0 restart' to restart."
            exit 1
        fi
        nohup $SCRIPT > "$LOG_FILE" 2>&1 &
        save_pid $!
        echo "✅ Bot started (PID: $(get_pid))"
        echo "📜 Logs: $LOG_FILE"
        echo "💡 To monitor: $0 monitor"
        ;;
    
    stop)
        echo "🛑 Stopping Nifty Scalper Bot..."
        if is_running; then
            kill "$(get_pid)" && sleep 2
            if is_running; then
                echo "⚠️ Forcing shutdown..."
                kill -9 "$(get_pid)" 2>/dev/null || true
            fi
            clear_pid
            echo "✅ Stopped."
        else
            echo "💤 Not running."
        fi
        ;;
    
    restart)
        echo "🔄 Restarting..."
        "$0" stop
        sleep 2
        "$0" start
        ;;
    
    status)
        echo "📊 Status:"
        if is_running; then
            echo "  ✅ RUNNING (PID: $(get_pid))"
        else
            echo "  ❌ STOPPED"
            [[ -f "$PID_FILE" ]] && echo "  ⚠️ Stale PID: $(cat $PID_FILE)" || echo "  ℹ️ No PID file."
        fi
        ;;
    
    logs)
        echo "📄 Last 20 log lines:"
        [[ -f "$LOG_FILE" ]] && tail -n 20 "$LOG_FILE" || echo "⚠️ Log file missing."
        ;;
    
    monitor)
        echo "🔍 Live Logs (Ctrl+C to exit)"
        [[ ! -f "$LOG_FILE" ]] && touch "$LOG_FILE"
        tail -f "$LOG_FILE" &
        TAIL_PID=$!
        trap 'kill $TAIL_PID > /dev/null 2>&1; exit 0' INT TERM
        wait $TAIL_PID
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|monitor}"
        echo "Examples:"
        echo "  $0 start     # Start bot"
        echo "  $0 stop      # Stop bot"
        echo "  $0 status    # Check if running"
        echo "  $0 logs      # Show last 20 lines"
        echo "  $0 monitor   # Follow logs live"
        exit 1
        ;;
esac
