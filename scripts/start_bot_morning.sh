#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
python -m nifty_scalper_bot.app
