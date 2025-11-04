#!/usr/bin/env bash
set -euo pipefail
ruff check src
if [ -d tests/diagnostics ]; then
  ruff check tests/diagnostics
fi
mypy \
  src/nifty_scalper_bot/diagnostics/assess.py \
  src/nifty_scalper_bot/data/assess_data.py \
  src/nifty_scalper_bot/risk/assess_risk.py \
  src/nifty_scalper_bot/execution/assess_orders.py \
  src/nifty_scalper_bot/server/selftest.py \
  src/nifty_scalper_bot/utils/invariants.py
if [ -d tests/diagnostics ]; then
  pytest tests/diagnostics -q
else
  pytest -q
fi
# optional: curl selftest if server is running
# curl -sS http://localhost:8080/selftest | jq
