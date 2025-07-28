#!/bin/bash

echo "🧹 Cleaning and Starting Nifty Scalper Bot..."
echo "=========================================="

# Stop all existing bot processes
echo "🛑 Stopping existing bot processes..."
pkill -f "src/main.py" 2>/dev/null
pkill -f "web_dashboard" 2>/dev/null
pkill -f "websocket_client" 2>/dev/null
pkill -f "telegram_controller" 2>/dev/null

# Wait for processes to stop
sleep 3

# Check if any processes are still running
echo "🔍 Checking for remaining processes..."
if pgrep -f "src/main.py" > /dev/null; then
    echo "⚠️  Some processes still running. Force killing..."
    pkill -9 -f "src/main.py" 2>/dev/null
fi

# Clear logs
echo "🗑️  Clearing old logs..."
rm -f logs/*.log 2>/dev/null

# Create fresh logs directory
mkdir -p logs

# Start bot in background
echo "🚀 Starting bot in background..."
nohup python src/main.py --mode realtime --trade > logs/bot.log 2>&1 &

# Capture PID
BOT_PID=$!

echo "✅ Bot started with PID: $BOT_PID"
echo ""
echo "📊 Monitor logs: tail -f logs/bot.log"
echo "🛑 Stop bot: ./process_manager.sh stop"
echo "📱 Control via Telegram commands"
echo "🌐 Web dashboard: http://localhost:8000 (if running)"

# Wait a moment for startup
sleep 5

# Check if bot is running
if ps -p $BOT_PID > /dev/null; then
    echo ""
    echo "🎉 Bot is running successfully!"
    echo "🕐 $(date '+%Y-%m-%d %H:%M:%S %Z')"
else
    echo ""
    echo "❌ Bot failed to start. Check logs:"
    tail -n 20 logs/bot.log
fi
