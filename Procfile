web: cd /app && set -a && [ -f .env ] && source .env && set +a && python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port ${PORT:-8000}
