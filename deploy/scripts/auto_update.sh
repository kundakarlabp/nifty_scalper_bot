#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
STATUS_FILE="${APP_DIR}/data/auto_update_status.txt"
PENDING_FILE="${APP_DIR}/data/bot_restart_pending"
mkdir -p "${APP_DIR}/data"
cd "${APP_DIR}"

status() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" > "${STATUS_FILE}"
}

exec 9>/tmp/niftybot-auto-update.lock
flock -n 9 || exit 0

if ! git diff --quiet || ! git diff --cached --quiet; then
  status "Skipped: uncommitted repository changes"
  exit 0
fi

git fetch --quiet origin main
LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse origin/main)"

if [[ "${LOCAL}" != "${REMOTE}" ]]; then
  OLD="${LOCAL}"
  git merge --ff-only origin/main
  if ! python3 -m compileall -q dashboard src; then
    git reset --hard "${OLD}"
    status "Update rolled back: validation failed"
    exit 1
  fi
  sudo systemctl restart niftybot-streamlit.service
  touch "${PENDING_FILE}"
  status "Updated dashboard; bot restart queued outside market hours"
fi

NOW="$(TZ=Asia/Kolkata date +%H%M)"
if [[ -f "${PENDING_FILE}" ]] && (( 10#${NOW} < 845 || 10#${NOW} > 1545 )); then
  sudo systemctl restart niftybot.service
  rm -f "${PENDING_FILE}"
  status "Dashboard and bot updated to $(git rev-parse --short HEAD)"
elif [[ "${LOCAL}" == "${REMOTE}" ]] && [[ ! -f "${PENDING_FILE}" ]]; then
  status "Up to date at $(git rev-parse --short HEAD)"
fi
