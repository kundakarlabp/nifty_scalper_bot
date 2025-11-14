#!/usr/bin/env bash
set -euo pipefail

_term() {
  kill -TERM "$child" 2>/dev/null || true
}

trap _term SIGTERM

export PYTHONPATH="${PYTHONPATH:-/app/src}"
python -m nifty_scalper_bot.app &
child=$!
wait "$child"
