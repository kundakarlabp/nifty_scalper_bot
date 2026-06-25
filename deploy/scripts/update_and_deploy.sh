#!/usr/bin/env bash
# File purpose: Backward-compatible entrypoint for an AWS Lightsail deployment.
# Key responsibilities: Delegate all updates to the canonical validated release runner.
# Operational constraints: Never embed credentials; never bypass candidate validation or readiness checks.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-$HOME/nifty_scalper_bot}"
RELEASE_SCRIPT="$APP_DIR/deploy/lightsail_release.sh"

if [ ! -x "$RELEASE_SCRIPT" ]; then
  printf 'ERROR: canonical Lightsail release script not found or not executable: %s\n' "$RELEASE_SCRIPT" >&2
  exit 1
fi

exec "$RELEASE_SCRIPT" --force
