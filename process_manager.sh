#!/bin/bash

ACTION=${1:-status}

case $ACTION in
    status)
        echo "📊 Nifty Scalper Bot Status"
        echo "=========================="
        
        # Check main bot
        if pgrep -f "src/main.py" > /dev/null; then
            echo "✅ Main Bot: RUNNING"
            echo "   PIDs: $(pgrep -f "src/main.py")"
        else
            echo "❌ Main Bot: STOPPED"
        fi
        
        # Check web dashboard
        if pgrep -f "web_dashboard" > /dev/null; then
            echo "✅ Web Dashboard: RUNNING"
            echo "   PIDs: $(pgrep -f "web_dashboard")"
        else
            echo "❌ Web Dashboard: STOPPED"
        fi
        
        # Check logs
        if [ -d "logs" ]; then
            echo "📁 Logs: EXISTS ($(ls logs/*.log 2>/dev/null | wc -l) files)"
        else
            echo "❌ Logs: DIRECTORY MISSING"
        fi
        
        echo ""
        echo "🔧 Commands:"
        echo "   ./process_manager.sh start    # Start bot"
        echo "   ./process_manager.sh stop     # Stop bot"
        echo "   ./process_manager.sh restart  # Restart bot"
        echo "   ./process_manager.sh monitor  # Monitor logs"
        echo "   ./process_manager.sh status   # Show status"
        ;;
        
    start)
        echo "🚀 Starting Nifty Scalper Bot..."
        
        # Create logs directory
        mkdir -p logs
        
        # Stop any existing processes first
        pkill -f "src/main.py" 2>/dev/null
        sleep 2
        
        # Start bot in background
        nohup python src/main.py --mode realtime --trade > logs/bot.log 2>&1 &
        BOT_PID=$!
        
        echo "✅ Bot started with PID: $BOT_PID"
        echo "📊 Logs: tail -f logs/bot.log"
        echo "🛑 To stop: ./process_manager.sh stop"
        ;;
        
    stop)
        echo "🛑 Stopping Nifty Scalper Bot..."
        
        # Stop all bot processes
        pkill -f "src/main.py" 2>/dev/null
        pkill -f "web_dashboard" 2>/dev/null
        pkill -f "websocket_client" 2>/dev/null
        pkill -f "telegram_controller" 2>/dev/null
        
        echo "✅ All bot processes stopped"
        ;;
        
    restart)
        echo "🔄 Restarting Nifty Scalper Bot..."
        
        # Stop existing processes
        pkill -f "src/main.py" 2>/dev/null
        sleep 3
        
        # Start bot in background
        mkdir -p logs
        nohup python src/main.py --mode realtime --trade > logs/bot.log 2>&1 &
        BOT_PID=$!
        
        echo "✅ Bot restarted with PID: $BOT_PID"
        echo "📊 Logs: tail -f logs/bot.log"
        ;;
        
    monitor)
        echo "🔍 Monitoring Bot Logs (Press Ctrl+C to stop)..."
        echo "================================================"
        
        if [ -f "logs/bot.log" ]; then
            tail -f logs/bot.log
        else
            echo "❌ Log file not found: logs/bot.log"
        fi
        ;;
        
    *)
        echo "Usage: $0 {status|start|stop|restart|monitor}"
        echo ""
        echo "Commands:"
        echo "  status   - Show bot status"
        echo "  start    - Start bot in background"
        echo "  stop     - Stop all bot processes"
        echo "  restart  - Restart bot"
        echo "  monitor  - Monitor logs in real-time"
        ;;
esac
