#!/bin/bash
set -e

echo "================================================="
echo "🔵 NIFTY SCALPER BOT | $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================="

# === Change to /app directory ===
cd /app || { echo "❌ FATAL: Cannot cd to /app"; exit 1; }

# === Load .env file ===
echo "📂 Loading Environment..."
if [ -f "/app/.env" ]; then
    echo "   ✅ Found /app/.env"
    set -a
    source /app/.env
    set +a
elif [ -f "./.env" ]; then
    echo "   ✅ Found ./.env"
    set -a
    source ./.env
    set +a
else
    echo "   ⚠️ No .env file - using system environment"
fi

# === FIX: Ensure data directories are writable ===
echo "📂 Setting up data directories..."

# Check if /app/data exists and is writable
if [ -d "/app/data" ]; then
    if touch /app/data/.write_test 2>/dev/null; then
        rm -f /app/data/.write_test
        export DATA_DIR="/app/data"
        echo "   ✅ Using /app/data (volume mounted, writable)"
    else
        echo "   ⚠️ /app/data exists but NOT writable (permission denied)"
        export DATA_DIR="/tmp/nifty_scalper_data"
        mkdir -p "$DATA_DIR"
        echo "   ✅ Using fallback: $DATA_DIR"
    fi
else
    # /app/data doesn't exist, use fallback
    export DATA_DIR="/tmp/nifty_scalper_data"
    mkdir -p "$DATA_DIR"
    echo "   ✅ Using fallback: $DATA_DIR"
fi

# Also ensure ./data exists for relative path code
mkdir -p ./data 2>/dev/null || true
if touch ./data/.write_test 2>/dev/null; then
    rm -f ./data/.write_test
    echo "   ✅ ./data is writable"
else
    echo "   ⚠️ ./data not writable, symlinking to DATA_DIR"
    rm -rf ./data 2>/dev/null || true
    ln -sf "$DATA_DIR" ./data 2>/dev/null || true
fi

echo "   DATA_DIR=$DATA_DIR"

# === Environment Check ===
echo "🔍 Environment Check:"
echo "   PORT=${PORT:-8000}"
echo "   ENABLE_LIVE=${ENABLE_LIVE:-NOT_SET}"
echo "   EXECUTION_MODE=${EXECUTION_MODE:-NOT_SET}"
echo "   DATA_DIR=${DATA_DIR}"

# === Set defaults ===
export PORT=${PORT:-8000}

# === Warning for SHADOW mode ===
if [ "$ENABLE_LIVE" != "true" ] && [ "$ENABLE_LIVE" != "True" ] && [ "$ENABLE_LIVE" != "TRUE" ]; then
    echo ""
    echo "⚠️  WARNING: ENABLE_LIVE is NOT 'true'"
    echo "   Current value: '${ENABLE_LIVE:-NOT_SET}'"
    echo "   Bot will run in SHADOW mode - NO REAL TRADES!"
    echo ""
fi

# === Launch ===
echo "================================================="
echo "🚀 LAUNCHING UVICORN on port $PORT"
echo "================================================="

exec python -m uvicorn nifty_scalper_bot.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
