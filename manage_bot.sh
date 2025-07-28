#!/bin/bash

ACTION=${1:-status}

case $ACTION in
    start)
        echo "🚀 Starting Nifty Scalper Bot..."
        mkdir -p logs
        
        # Start trading bot in background
        nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &
        TRADING_BOT_PID=$!
        echo "✅ Trading bot started with PID: $TRADING_BOT_PID"
        
        # Start Telegram command listener in background
        nohup python telegram_command_listener.py > logs/telegram_listener.log 2>&1 &
        TELEGRAM_BOT_PID=$!
        echo "✅ Telegram command listener started with PID: $TELEGRAM_BOT_PID"
        
        echo "📊 Logs:"
        echo "   Trading Bot: tail -f logs/trading_bot.log"
        echo "   Telegram Listener: tail -f logs/telegram_listener.log"
        echo "🛑 To stop: ./manage_bot.sh stop"
        ;;
        
    stop)
        echo "🛑 Stopping Nifty Scalper Bot..."
        
        # Stop trading bot
        pkill -f "src/main.py"
        
        # Stop Telegram listener
        pkill -f "telegram_command_listener.py"
        
        echo "✅ All bot processes stopped"
        ;;
        
    restart)
        echo "🔄 Restarting Nifty Scalper Bot..."
        
        # Stop existing processes
        pkill -f "src/main.py"
        pkill -f "telegram_command_listener.py"
        sleep 3
        
        # Start new processes
        mkdir -p logs
        nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &
        TRADING_BOT_PID=$!
        nohup python telegram_command_listener.py > logs/telegram_listener.log 2>&1 &
        TELEGRAM_BOT_PID=$!
        
        echo "✅ Bot restarted"
        echo "   Trading Bot PID: $TRADING_BOT_PID"
        echo "   Telegram Listener PID: $TELEGRAM_BOT_PID"
        ;;
        
    status)
        echo "📊 Nifty Scalper Bot Status:"
        
        # Check trading bot
        if pgrep -f "src/main.py" > /dev/null; then
            echo "   Trading Bot: ✅ RUNNING"
            echo "   PID: $(pgrep -f "src/main.py")"
        else
            echo "   Trading Bot: ❌ STOPPED"
        fi
        
        # Check Telegram listener
        if pgrep -f "telegram_command_listener.py" > /dev/null; then
            echo "   Telegram Listener: ✅ RUNNING"
            echo "   PID: $(pgrep -f "telegram_command_listener.py")"
        else
            echo "   Telegram Listener: ❌ STOPPED"
        fi
        
        echo "   Logs Directory: logs/"
        echo "   Trading Logs: logs/trading_bot.log"
        echo "   Telegram Logs: logs/telegram_listener.log"
        ;;
        
    logs)
        echo "📜 Recent Bot Logs:"
        echo ""
        echo "=== Trading Bot Logs ==="
        tail -n 20 logs/trading_bot.log 2>&1 || echo "No trading bot logs available"
        echo ""
        echo "=== Telegram Listener Logs ==="
        tail -n 20 logs/telegram_listener.log 2>&1 || echo "No Telegram listener logs available"
        ;;
        
    monitor)
        echo "🔍 Monitoring Bot (Press Ctrl+C to stop)..."
        echo ""
        echo "=== Trading Bot Logs ==="
        tail -f logs/trading_bot.log 2>&1 &
        TAIL_PID1=$!
        echo ""
        echo "=== Telegram Listener Logs ==="
        tail -f logs/telegram_listener.log 2>&1 &
        TAIL_PID2=$!
        
        # Trap Ctrl+C to kill tail processes
        trap "kill $TAIL_PID1 $TAIL_PID2; exit" INT
        wait
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
