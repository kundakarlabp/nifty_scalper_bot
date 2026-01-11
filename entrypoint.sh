#!/bin/bash
set -e

# --- 1. LIFE SIGNS ---
echo "================================================="
echo "🔵 CONTAINER BOOT: $(date)"
echo "================================================="

# --- 2. DIAGNOSTICS ---
echo "🔍 DIAGNOSTIC: Checking File Structure..."
ls -la /app

echo "🔍 DIAGNOSTIC: Checking Environment..."
if [ -z "$KITE_API_KEY" ]; then echo "❌ KITE_API_KEY: MISSING"; else echo "✅ KITE_API_KEY: FOUND"; fi
if [ -z "$PORT" ]; then 
    echo "⚠️ PORT var missing. Defaulting to 8000"
    export PORT=8000
else
    echo "✅ PORT: $PORT"
fi

# --- 3. LAUNCH ---
echo "🚀 STARTING UVICORN..."
# 'exec' ensures the app receives shutdown signals correctly
exec uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port "$PORT" --log-level info
