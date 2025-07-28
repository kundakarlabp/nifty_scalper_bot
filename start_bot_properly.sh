#!/bin/bash

echo "🚀 Starting Nifty Scalper Bot Properly..."

# Stop any existing bot processes
echo "🛑 Stopping existing bot processes..."
pkill -f 'src/main.py' 2>/dev/null
sleep 3

# Create logs directory
mkdir -p logs

# Run the bot in background with proper logging
echo "🔧 Starting bot in background..."
nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &

# Capture the process ID
BOT_PID=$!
echo "✅ Trading bot started with PID: $BOT_PID"

# Wait a moment for startup
sleep 5

# Check if bot is running
if ps -p $BOT_PID > /dev/null; then
    echo "✅ Bot is running successfully!"
    echo ""
    echo "📊 Monitor logs: tail -f logs/trading_bot.log"
    echo "🛑 Stop bot: kill $BOT_PID"
    echo "📱 Control via Telegram commands:"
    echo "   /start - Start trading system"
    echo "   /stop - Stop trading system"  
    echo "   /status - Show system status"
    echo "   /help - Show all commands"
    echo "   /enable - Enable trade execution"
    echo "   /disable - Disable trade execution"
else
    echo "❌ Bot failed to start. Check logs:"
    tail -n 20 logs/trading_bot.log
fi
