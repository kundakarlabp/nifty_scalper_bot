#!/bin/bash
set -e

echo "================================================="
echo "🔵 NIFTY SCALPER BOT v2.0 | $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================="

# === CRITICAL: Change to /app first ===
cd /app || { echo "❌ FATAL: Cannot cd to /app"; exit 1; }

# === ENVIRONMENT LOADING ===
echo "📂 Loading Environment..."

ENV_LOADED=false

# Try multiple .env locations
for env_file in "/app/.env" "./.env" "/home/appuser/.env"; do
    if [ -f "$env_file" ]; then
        echo "   ✅ Found: $env_file"
        set -a
        source "$env_file"
        set +a
        ENV_LOADED=true
        break
    fi
done

if [ "$ENV_LOADED" = false ]; then
    echo "   ⚠️ No .env file found - using Railway environment variables only"
fi

# === ENVIRONMENT VALIDATION ===
echo "🔍 Environment Check:"
echo "   PORT=${PORT:-8000}"
echo "   ENABLE_LIVE=${ENABLE_LIVE:-NOT_SET}"
echo "   EXECUTION_MODE=${EXECUTION_MODE:-NOT_SET}"
echo "   PWD=$(pwd)"

# Set defaults
export PORT=${PORT:-8000}

# === CRITICAL VARIABLE WARNINGS ===
if [ "$ENABLE_LIVE" != "true" ] && [ "$ENABLE_LIVE" != "True" ] && [ "$ENABLE_LIVE" != "TRUE" ]; then
    echo ""
    echo "⚠️  WARNING: ENABLE_LIVE is NOT 'true'"
    echo "   Current value: '${ENABLE_LIVE:-NOT_SET}'"
    echo "   Bot will run in SHADOW mode - NO REAL TRADES!"
    echo ""
fi

# === QUICK IMPORT TEST ===
echo "🔧 Verifying Python imports..."
python -c "from nifty_scalper_bot.main import app; print('   ✅ Main module OK')" 2>&1 || {
    echo "❌ FATAL: Cannot import main module"
    echo "Running diagnostic..."
    python -c "
import sys
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')
try:
    import nifty_scalper_bot
except Exception as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
"
    exit 1
}

# === LAUNCH ===
echo "================================================="
echo "🚀 LAUNCHING UVICORN on port $PORT"
echo "================================================="

exec python -m uvicorn nifty_scalper_bot.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --access-log
