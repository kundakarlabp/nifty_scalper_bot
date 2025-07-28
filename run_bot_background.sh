#!/bin/bash

echo "🚀 Running Nifty Scalper Bot in Background..."

# Create logs directory
mkdir -p logs

# Run the bot in background with nohup
nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &

# Capture the process ID
BOT_PID=$!

echo "✅ Trading bot started with PID: $BOT_PID"
echo "📊 Logs are being written to: logs/trading_bot.log"
echo "🔧 To monitor logs: tail -f logs/trading_bot.log"
echo "🛑 To stop the bot: kill $BOT_PID"
echo "📱 You can now use Telegram commands to control the bot!"

# Wait a moment for startup
sleep 5

# Check if bot is running
if ps -p $BOT_PID > /dev/null; then
    echo "✅ Bot is running successfully in background!"
else
    echo "❌ Bot failed to start. Check logs:"
    tail -n 20 logs/trading_bot.log
fi
