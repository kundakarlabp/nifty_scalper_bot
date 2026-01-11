#!/bin/bash
set -e

# --- 1. LIFE SIGNS ---
echo "================================================="
echo "🔵 CONTAINER BOOT: $(date)"
echo "================================================="

# --- 2. ENVIRONMENT CHECK ---
echo "🔍 Checking Environment..."
if [ -z "$KITE_API_KEY" ]; then echo "❌ KITE_API_KEY: MISSING"; else echo "✅ KITE_API_KEY: FOUND"; fi

if [ -z "$PORT" ]; then 
    echo "⚠️ PORT variable missing. Defaulting to 8000"
    export PORT=8000
else
    echo "✅ PORT detected: $PORT"
fi

# --- 3. LAUNCH ---
echo "🚀 LAUNCHING APP..."
# USE 'python -m uvicorn' INSTEAD OF JUST 'uvicorn'
exec python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port "$PORT" --log-level info
