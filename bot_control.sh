#!/bin/bash

ACTION=${1:-status}

case $ACTION in
    start)
        echo "🚀 Starting Nifty Scalper Bot..."
        mkdir -p logs
        nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &
        BOT_PID=$!
        echo "✅ Bot started with PID: $BOT_PID"
        echo "📊 Logs: tail -f logs/trading_bot.log"
        ;;
    stop)
        echo "🛑 Stopping Nifty Scalper Bot..."
        pkill -f "src/main.py"
        echo "✅ Bot stopped"
        ;;
    restart)
        echo "🔄 Restarting Nifty Scalper Bot..."
        pkill -f "src/main.py"
        sleep 3
        mkdir -p logs
        nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &
        BOT_PID=$!
        echo "✅ Bot restarted with PID: $BOT_PID"
        ;;
    status)
        echo "📊 Nifty Scalper Bot Status:"
        if pgrep -f "src/main.py" > /dev/null; then
            echo "   Status: ✅ RUNNING"
            echo "   PID: $(pgrep -f "src/main.py")"
        else
            echo "   Status: ❌ STOPPED"
        fi
        echo "   Logs: logs/trading_bot.log"
        ;;
    logs)
        echo "📜 Recent Bot Logs:"
        tail -n 50 logs/trading_bot.log 2>&1 || echo "No logs available"
        ;;
    monitor)
        echo "🔍 Monitoring Bot (Press Ctrl+C to stop)..."
        tail -f logs/trading_bot.log 2>&1 || echo "No logs available"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|monitor}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the bot"
        echo "  stop      - Stop the bot"
        echo "  restart   - Restart the bot"
        echo "  status    - Show bot status"
        echo "  logs      - Show recent logs"
        echo "  monitor   - Monitor logs in real-time"
        ;;
esac
