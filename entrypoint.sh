#!/bin/sh
set -e

# 1. Immediate Life Sign
echo "================================================="
echo "🔵 CONTAINER BOOT SEQUENCE INITIATED"
echo "🔵 Timestamp: $(date)"
echo "================================================="

# 2. Environment Check (Crucial for "Immediate Crash")
echo "🔍 Checking Critical Environment Variables..."
if [ -z "$KITE_API_KEY" ]; then echo "❌ KITE_API_KEY is MISSING"; else echo "✅ KITE_API_KEY is set"; fi
if [ -z "$KITE_API_SECRET" ]; then echo "❌ KITE_API_SECRET is MISSING"; else echo "✅ KITE_API_SECRET is set"; fi
if [ -z "$KITE_ACCESS_TOKEN" ]; then echo "❌ KITE_ACCESS_TOKEN is MISSING"; else echo "✅ KITE_ACCESS_TOKEN is set"; fi

# 3. Directory Check
echo "🔍 Current Directory: $(pwd)"
echo "🔍 Files in root:"
ls -la

# 4. Start Python
echo "🚀 LAUNCHING UVICORN..."
echo "================================================="

# Force Unbuffered output so logs appear instantly
export PYTHONUNBUFFERED=1

# Execute the app
exec uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port $PORT --log-level info
