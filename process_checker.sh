#!/bin/bash

echo "🔍 Nifty Scalper Bot Process Checker"
echo "==================================="

# Function to check if process is running
check_process() {
    local process_name=$1
    local process_count=$(ps aux | grep "$process_name" | grep -v grep | wc -l)
    
    if [ $process_count -gt 0 ]; then
        echo "✅ $process_name: RUNNING ($process_count instances)"
        ps aux | grep "$process_name" | grep -v grep | head -5
    else
        echo "❌ $process_name: NOT RUNNING"
    fi
}

# Check main bot processes
echo "1. Checking Main Bot Processes..."
check_process "src/main.py"
echo ""

# Check web dashboard
echo "2. Checking Web Dashboard..."
check_process "web_dashboard"
echo ""

# Check WebSocket client
echo "3. Checking WebSocket Client..."
check_process "websocket_client"
echo ""

# Check Telegram controller
echo "4. Checking Telegram Controller..."
check_process "telegram_controller"
echo ""

# Check all Python processes
echo "5. Checking All Python Processes..."
python_count=$(ps aux | grep "python" | grep -v grep | wc -l)
echo "🐍 Total Python processes: $python_count"
ps aux | grep "python" | grep -v grep | grep -v "grep" | head -10
echo ""

# Check log files
echo "6. Checking Log Files..."
if [ -d "logs" ]; then
    echo "📁 Log directory exists"
    log_files=$(ls -la logs/ 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "📄 Log files:"
        ls -la logs/
        echo ""
        echo "📊 Recent log entries:"
        tail -n 10 logs/*.log 2>/dev/null | head -20
    else
        echo "⚠️  No log files found"
    fi
else
    echo "❌ Log directory does not exist"
fi
echo ""

# Check resource usage
echo "7. Checking Resource Usage..."
echo "💾 Memory usage:"
free -h
echo ""
echo "📈 CPU usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo ""
echo "使用網路 usage:"
df -h /
echo ""

echo "🎉 Process check completed!"
echo "🔧 Useful commands:"
echo "   ps aux | grep 'src/main.py'     # Check bot processes"
echo "   kill -9 <PID>                  # Kill specific process"
echo "   pkill -f 'src/main.py'         # Kill all bot processes"
echo "   tail -f logs/*.log             # Monitor logs"
echo "   nohup python src/main.py --mode realtime --trade > logs/bot.log 2>&1 &  # Start in background"
