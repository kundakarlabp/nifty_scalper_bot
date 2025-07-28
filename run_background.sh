#!/bin/bash

echo "🚀 Starting Nifty Scalper Bot in Background..."
echo "=========================================="

# Create logs directory
mkdir -p logs

# Run the bot in background with nohup
echo "🔧 Starting trading bot in background..."
nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &

# Capture the process ID
BOT_PID=$!

echo "✅ Trading bot started with PID: $BOT_PID"
echo "📊 Logs are being written to: logs/trading_bot.log"
echo ""
echo "🔧 Useful Commands:"
echo "   🔍 Monitor logs: tail -f logs/trading_bot.log"
echo "   �� View recent logs: tail -n 50 logs/trading_bot.log"
echo "   🚫 Stop bot: pkill -f 'src/main.py'"
echo "   🔄 Restart bot: ./run_background.sh"
echo "   📊 Check status: ps aux | grep 'src/main.py'"
echo ""
echo "📱 You can now control the bot via Telegram commands!"
echo "   /start - Start trading system"
echo "   /stop - Stop trading system"
echo "   /status - Show system status"
echo "   /help - Show all commands"
echo ""
echo "🌐 Web dashboard: http://localhost:8000 (when running)"
