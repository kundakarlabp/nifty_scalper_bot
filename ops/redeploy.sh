#!/usr/bin/env bash
# AWS Lightsail redeploy shortcut. Delegates to the validated release runner.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
RELEASE_SCRIPT="$APP_DIR/deploy/lightsail_release.sh"

if [ ! -x "$RELEASE_SCRIPT" ]; then
  printf 'ERROR: release script unavailable: %s\n' "$RELEASE_SCRIPT" >&2
  exit 1
fi

exec "$RELEASE_SCRIPT" --force
