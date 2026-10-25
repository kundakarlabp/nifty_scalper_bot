#!/usr/bin/env bash
set -Eeuo pipefail

# --- basic env / paths ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
if [[ -d "$ROOT_DIR/src" ]]; then
  export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-$ROOT_DIR/src}"
else
  export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"
fi

# Never shell-source a repository .env: it may execute arbitrary commands
# and can override the protected host policy. Python settings resolve the one
# canonical BOT_ENV_FILE in production and local dotenv in development.

cmd="${1:-run}"

log() { printf '[manage_bot] %s\n' "$*"; }

# quick sanity for pydantic v2 layout
sanity_check() {
  python - <<'PY'
from pydantic_settings import BaseSettings  # noqa
print("pydantic-settings OK")
PY
}

start_trader() {
  log "Starting trader (shadow mode by default)"
  exec python -u -m nifty_scalper_bot.main
}

run_backtest() {
  FROM="${1:-2024-01-01}"
  TO="${2:-2024-01-05}"
  log "Backtest ${FROM} → ${TO}"
  exec python -m nifty_scalper_bot.backtesting.engine --from "$FROM" --to "$TO" --generate-sample
}

token_helper() {
  # A manual SSH shell does not necessarily inherit systemd's environment.
  # Never fall back to a repository .env, even when the platform flag is absent.
  if [[ -z "${BOT_ENV_FILE:-}" || "${BOT_ENV_FILE}" != /* ]]; then
    echo "Token helper requires an absolute BOT_ENV_FILE; use the authenticated admin page otherwise." >&2
    exit 2
  fi
  log "Launching Zerodha token helper"
  exec python scripts/zerodha_token_cli.py --env "$BOT_ENV_FILE" --write-env
}

case "$cmd" in
  run)             sanity_check; start_trader ;;
  backtest)        shift; run_backtest "$@" ;;
  token)           token_helper ;;
  *)
    echo "Usage: bash manage_bot.sh [run|backtest <FROM> <TO>|token]" >&2
    exit 2
    ;;
esac
