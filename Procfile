web: cd /app && set -a && [ -f .env ] && source .env && set +a && python -m uvicorn nifty_scalper_bot.deployment_main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-1}
