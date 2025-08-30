#!/usr/bin/env bash
set -euo pipefail

_term() {
  kill -TERM "$child" 2>/dev/null || true
}

trap _term SIGTERM

python -m src.main start &
child=$!
wait "$child"
