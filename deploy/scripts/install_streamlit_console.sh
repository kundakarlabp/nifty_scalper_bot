#!/usr/bin/env bash
# File purpose: Install or refresh the independent admin and read-only review controls.
# Key responsibilities: Preserve external settings, validate imports, install the bounded service, and verify both ports.
# Operational constraints: Never rewrite existing credentials; keep control requests fast and isolated from the trading loop.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
VENV_DIR="${APP_DIR}/.streamlit-venv"
ENGINE_VENV="${APP_DIR}/.venv"
SERVICE_SOURCE="${APP_DIR}/deploy/systemd/niftybot-streamlit.service"
SERVICE_TARGET="/etc/systemd/system/niftybot-streamlit.service"
ENV_FILE="${BOT_ENV_FILE:-/home/ubuntu/.config/niftybot/niftybot.env}"

ensure_default() {
  local key="$1" value="$2"
  grep -qE "^${key}=" "$ENV_FILE" || printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
}

cd "${APP_DIR}"
mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"
ensure_default POST_MARKET_QUIET_MODE true
ensure_default POST_MARKET_BASKET_REFRESH_SECONDS 1800
ensure_default POST_MARKET_MARKET_DATA_SUMMARY_SECONDS 900
ensure_default POST_MARKET_SUPPRESS_CANDLE_GAP_WARNINGS true
ensure_default BOT_ADMIN_API_URL http://127.0.0.1:8081
ensure_default BOT_ADMIN_PUBLIC_URL http://15.206.3.6:8081/admin

if ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# Remove abandoned pip-rename metadata that produces the repeated
# "Ignoring invalid distribution ~ifty-scalper-bot" warning.
if [[ -x "${ENGINE_VENV}/bin/python" ]]; then
  SITE="$(${ENGINE_VENV}/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  find "$SITE" -maxdepth 1 -name '~*' -exec rm -rf -- {} + 2>/dev/null || true
fi

PIP_NO_CACHE_DIR=1 "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
PIP_NO_CACHE_DIR=1 "${VENV_DIR}/bin/python" -m pip install --quiet -r dashboard/requirements.txt

PYTHONPATH="${APP_DIR}:${APP_DIR}/src" "${VENV_DIR}/bin/python" -m py_compile \
  dashboard/superlite_events.py \
  dashboard/superlite_console.py \
  dashboard/log_export.py
PYTHONPATH="${APP_DIR}/src:${APP_DIR}" "${ENGINE_VENV}/bin/python" -m py_compile \
  src/nifty_scalper_bot/superlite_admin_core.py \
  src/nifty_scalper_bot/superlite_admin.py
PYTHONPATH="${APP_DIR}:${APP_DIR}/src" "${VENV_DIR}/bin/python" -c \
  'from dashboard.superlite_events import parse_event; assert callable(parse_event)'
PYTHONPATH="${APP_DIR}/src:${APP_DIR}" "${ENGINE_VENV}/bin/python" -c \
  'from nifty_scalper_bot.superlite_admin import app; assert app is not None'

sudo install -m 0644 "${SERVICE_SOURCE}" "${SERVICE_TARGET}"
sudo systemctl daemon-reload
sudo systemctl enable --quiet niftybot-streamlit.service
sudo systemctl enable --quiet --now niftybot-autodeploy.timer
sudo systemctl restart niftybot-streamlit.service

admin_ok=false
review_ok=false
for _ in $(seq 1 25); do
  curl -fsS --max-time 2 http://127.0.0.1:8081/healthz >/dev/null 2>&1 && admin_ok=true
  curl -fsS --max-time 2 http://127.0.0.1:8501/ >/dev/null 2>&1 && review_ok=true
  [[ "$admin_ok" == true && "$review_ok" == true ]] && break
  sleep 1
done

if [[ "$admin_ok" != true || "$review_ok" != true ]]; then
  sudo systemctl status niftybot-streamlit.service --no-pager -l || true
  exit 1
fi

echo "Admin controls ready: http://15.206.3.6:8081/admin"
echo "Read-only review ready: http://15.206.3.6:8501"
