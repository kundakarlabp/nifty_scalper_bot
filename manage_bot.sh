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

# Optional: load .env if present (Railway also injects envs)
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

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
  log "Launching Zerodha token helper"
  exec python scripts/zerodha_token_cli.py --write-env
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
