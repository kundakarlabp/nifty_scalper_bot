#!/bin/bash
set -e

echo "================================================="
echo "🔵 NIFTY SCALPER BOT STARTUP: $(date)"
echo "================================================="

# === ENVIRONMENT LOADING ===
echo "🔍 Loading Environment Variables..."

# Multiple .env locations for robustness
ENV_LOADED=false
for env_file in "/app/.env" "/.env" "./.env"; do
    if [ -f "$env_file" ]; then
        echo "✅ Found env file: $env_file"
        set -a
        source "$env_file"
        set +a
        ENV_LOADED=true
        break
    fi
done

if [ "$ENV_LOADED" = false ]; then
    echo "⚠️  WARNING: No .env file found - using system environment only"
fi

# === CRITICAL VARIABLE VALIDATION ===
echo "🔍 Validating Critical Trading Variables..."

# Trading Mode Validation
if [ "$ENABLE_LIVE" = "true" ] || [ "$ENABLE_LIVE" = "True" ] || [ "$ENABLE_LIVE" = "TRUE" ] || [ "$ENABLE_LIVE" = "1" ]; then
    echo "✅ TRADING MODE: LIVE (ENABLE_LIVE=$ENABLE_LIVE)"
    TRADING_ENABLED=true
else
    echo "⚠️  WARNING: SHADOW/PAPER MODE (ENABLE_LIVE=${ENABLE_LIVE:-NOT_SET})"
    TRADING_ENABLED=false
fi

# API Keys Validation
if [ -z "$ZERODHA_API_KEY" ] && [ -z "$KITE_API_KEY" ]; then
    echo "❌ CRITICAL: No API key found (need ZERODHA_API_KEY or KITE_API_KEY)"
    TRADING_ENABLED=false
else
    echo "✅ API KEY: Found"
fi

# Access Token Validation
if [ -z "$ZERODHA_ACCESS_TOKEN" ] && [ -z "$KITE_ACCESS_TOKEN" ]; then
    echo "❌ CRITICAL: No access token found (need ZERODHA_ACCESS_TOKEN or KITE_ACCESS_TOKEN)"
    TRADING_ENABLED=false
else
    echo "✅ ACCESS TOKEN: Found"
fi

# Telegram Bot Validation
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "⚠️  WARNING: TELEGRAM_BOT_TOKEN not set - notifications disabled"
else
    echo "✅ TELEGRAM BOT: Configured"
fi

# Port Configuration
if [ -z "$PORT" ]; then
    echo "⚠️  PORT not set, defaulting to 8000"
    export PORT=8000
else
    echo "✅ PORT: $PORT"
fi

# === FINAL VALIDATION ===
echo "================================================="
if [ "$TRADING_ENABLED" = true ]; then
    echo "🚀 BOT READY FOR LIVE TRADING"
else
    echo "⚠️  BOT WILL RUN IN SHADOW/PAPER MODE"
fi
echo "================================================="

# === LAUNCH APPLICATION ===
echo "🚀 Starting Nifty Scalper Bot..."
cd /app
exec python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port "$PORT" --log-level info
