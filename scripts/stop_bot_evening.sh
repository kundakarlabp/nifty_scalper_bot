#!/usr/bin/env bash
set -euo pipefail
pkill -f "python -m nifty_scalper_bot.app" || true
