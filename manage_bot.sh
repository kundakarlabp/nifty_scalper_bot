#!/bin/bash
# Simple helper to start and stop the scalper bot and its telegram listener.

ACTION=${1:-status}
TRADING_SCRIPT="-m src.main"

case "$ACTION" in
    start)
        echo "🚀 Starting Nifty Scalper Bot..."
        mkdir -p logs

        # Launch the unified trading engine (includes Telegram polling)
        nohup python3 -m src.main start > logs/scalper_bot.log 2>&1 &
        BOT_PID=$!
        echo "✅ Scalper bot started (PID: $BOT_PID)"

        echo -e "📜 Tail the logs with:\n  tail -f logs/scalper_bot.log"
        ;;
    stop)
        echo "🛑 Stopping Nifty Scalper Bot..."
        pkill -f "python3 -m src.main start" || true
        echo "✅ Bot process stopped."
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    status)
        echo "📊 Bot status:"
        if pgrep -f "python3 -m src.main start" >/dev/null; then
            echo "  Scalper bot: ✅ RUNNING"
        else
            echo "  Scalper bot: ❌ STOPPED"
        fi
        ;;
    logs)
        echo "=== Scalper Bot (last 20 lines) ==="
        tail -n 20 logs/scalper_bot.log 2>/dev/null || echo "No logs yet."
        ;;
    monitor)
        echo "🔍 Live log monitoring (Ctrl+C to exit)"
        tail -f logs/scalper_bot.log &
        TAIL=$!
        trap "kill $TAIL; exit" INT
        wait
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|monitor}"
        ;;
esac