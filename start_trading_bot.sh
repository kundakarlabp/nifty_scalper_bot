#!/bin/bash

# Nifty Scalper Bot Production Startup Script

echo "🚀 Starting Nifty Scalper Trading Bot in Production Mode..."
echo "======================================================"

# Create logs directory
mkdir -p logs

# Check if bot is already running
if pgrep -f "src/main.py" > /dev/null; then
    echo "⚠️  Bot is already running. Stopping existing instance..."
    pkill -f "src/main.py"
    sleep 5
fi

# Start the bot in background with nohup
echo "🔧 Starting trading bot in background..."
nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &

# Capture the process ID
BOT_PID=$!

echo "✅ Trading bot started with PID: $BOT_PID"
echo "📊 Logs are being written to: logs/trading_bot.log"
echo ""
echo "🔧 Useful Commands:"
echo "   🔍 Monitor logs: tail -f logs/trading_bot.log"
echo "   📋 View recent logs: tail -n 50 logs/trading_bot.log"
echo "   🚫 Stop bot: pkill -f 'src/main.py'"
echo "   🔄 Restart bot: ./start_trading_bot.sh"
echo "   📊 Check status: ps aux | grep 'src/main.py'"
echo ""
echo "📱 You can now control the bot via Telegram commands!"
echo "   /start - Start trading"
echo "   /stop - Stop trading"  
echo "   /status - Show status"
echo "   /enable - Enable execution"
echo "   /disable - Disable execution"
echo ""
echo "🌐 Web dashboard: http://localhost:8000 (when running)"
