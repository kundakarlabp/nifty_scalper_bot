#!/bin/bash
echo "🔍 WebSocket Monitor Started"

while true; do
    # Check if main bot is running
    if ! pgrep -f "src/main.py" > /dev/null; then
        echo "❌ Main bot not running. Please start it first."
        exit 1
    fi
    
    # Send /status via Telegram and capture response (you'll need to check Telegram manually)
    echo "📢 Please send '/status' to your Telegram bot now..."
    
    # Wait a bit for you to check
    sleep 10
    
    # Check logs for recent WebSocket status
    echo "📋 Recent WebSocket related log entries:"
    tail -n 20 logs/trading_bot.log | grep -i "websocket\|connected" | tail -n 5
    
    # Wait before next check
    echo "⏳ Waiting 60 seconds before next check..."
    sleep 60
done
