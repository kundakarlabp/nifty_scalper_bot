#!/usr/bin/env bash
set -Eeuo pipefail

# --- basic env / paths ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

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
  python -m src.main start
}

run_backtest() {
  FROM="${1:-2024-01-01}"
  TO="${2:-2024-01-05}"
  log "Backtest ${FROM} → ${TO}"
  exec python -m src.backtesting.engine --from "$FROM" --to "$TO" --generate-sample
}

token_helper() {
  log "Launching Zerodha token helper"
  exec python scripts/zerodha_token_cli.py --write-env
}

case "$cmd" in
  run)
    sanity_check
    backoff=1
    while true; do
      if start_trader; then
        backoff=1
      else
        log "Fatal during startup, will retry in ${backoff}s"
        sleep "$backoff"
        backoff=$((backoff*2))
        if [ "$backoff" -gt 60 ]; then backoff=60; fi
      fi
    done
    ;;
  backtest)        shift; run_backtest "$@" ;;
  token)           token_helper ;;
  *)
    echo "Usage: bash manage_bot.sh [run|backtest <FROM> <TO>|token]" >&2
    exit 2
    ;;
esac
