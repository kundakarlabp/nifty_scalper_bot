#!/bin/bash
set -e

# --- 1. LIFE SIGNS ---
echo "================================================="
echo "🔵 CONTAINER BOOT: $(date)"
echo "================================================="

# --- 2. LOAD .ENV FILE (CRITICAL FIX) ---
echo "🔍 Loading Environment..."

# Try multiple .env locations
if [ -f "/app/.env" ]; then
    echo "✅ Loading /app/.env"
    set -a
    source /app/.env
    set +a
elif [ -f "./.env" ]; then
    echo "✅ Loading ./.env"
    set -a
    source ./.env
    set +a
else
    echo "⚠️ No .env file found - using system environment only"
fi

# --- 3. ENVIRONMENT CHECK ---
echo "🔍 Verifying Critical Variables..."
echo "   ENABLE_LIVE=${ENABLE_LIVE:-NOT_SET}"
echo "   EXECUTION_MODE=${EXECUTION_MODE:-NOT_SET}"
echo "   FORCE_SIGNAL=${FORCE_SIGNAL:-NOT_SET}"

if [ -z "$ZERODHA_API_KEY" ] && [ -z "$KITE_API_KEY" ]; then 
    echo "❌ API KEY: MISSING (need ZERODHA_API_KEY or KITE_API_KEY)"
else 
    echo "✅ API KEY: FOUND"
fi

if [ -z "$PORT" ]; then 
    echo "⚠️ PORT variable missing. Defaulting to 8000"
    export PORT=8000
else
    echo "✅ PORT detected: $PORT"
fi

# --- 4. LAUNCH ---
echo "🚀 LAUNCHING APP..."
exec python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port "$PORT" --log-level info
