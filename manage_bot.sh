#!/bin/bash

ACTION=${1:-status}
BOT_SCRIPT="src/main.py"
TELEGRAM_SCRIPT="src/notifications/telegram_command_listener.py"

case $ACTION in
    start)
        echo "🚀 Starting Nifty Scalper Bot..."
        mkdir -p logs

        nohup python3 $BOT_SCRIPT --mode realtime --trade > logs/trading_bot.log 2>&1 &
        TRADING_BOT_PID=$!
        echo "✅ Trading bot started with PID: $TRADING_BOT_PID"

        nohup python3 $TELEGRAM_SCRIPT > logs/telegram_listener.log 2>&1 &
        TELEGRAM_BOT_PID=$!
        echo "✅ Telegram command listener started with PID: $TELEGRAM_BOT_PID"

        echo "📊 Logs:"
        echo "  tail -f logs/trading_bot.log"
        echo "  tail -f logs/telegram_listener.log"
        ;;
        
    stop)
        echo "🛑 Stopping Nifty Scalper Bot..."
        pkill -f "$BOT_SCRIPT"
        pkill -f "$TELEGRAM_SCRIPT"
        echo "✅ All processes stopped."
        ;;
        
    restart)
        echo "🔄 Restarting Bot..."
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo "📊 Bot Status:"
        pgrep -f "$BOT_SCRIPT" > /dev/null && echo "  Trading Bot: ✅ RUNNING" || echo "  Trading Bot: ❌ STOPPED"
        pgrep -f "$TELEGRAM_SCRIPT" > /dev/null && echo "  Telegram Bot: ✅ RUNNING" || echo "  Telegram Bot: ❌ STOPPED"
        ;;
        
    logs)
        echo "📜 Last 20 lines of logs:"
        echo "=== Trading Bot ==="
        tail -n 20 logs/trading_bot.log 2>/dev/null || echo "No trading logs yet."
        echo "=== Telegram Listener ==="
        tail -n 20 logs/telegram_listener.log 2>/dev/null || echo "No Telegram logs yet."
        ;;
        
    monitor)
        echo "🔍 Live log monitoring. Ctrl+C to stop."
        tail -f logs/trading_bot.log &
        TAIL1=$!
        tail -f logs/telegram_listener.log &
        TAIL2=$!
        trap "kill $TAIL1 $TAIL2; exit" INT
        wait
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|monitor}"
        ;;
esac