#!/bin/bash

# Nifty Scalper Web Dashboard Startup Script

echo "🚀 Starting Nifty Scalper Web Dashboard..."
echo "========================================"

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please navigate to src/web_dashboard directory"
    exit 1
fi

echo "✅ Found app.py in current directory"

# Test imports
echo "🧪 Testing imports..."
python -c "
import sys, os
sys.path.insert(0, os.getcwd())
try:
    from app import app
    print('✅ App imported successfully')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed"
    exit 1
fi

echo "🎉 All tests passed!"

echo ""
echo "🌐 Web Dashboard Starting..."
echo "   Visit: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

# Run the dashboard
python app.py
