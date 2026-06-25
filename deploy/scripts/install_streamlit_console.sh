#!/usr/bin/env bash
# File purpose: Install or refresh the lightweight Streamlit operations console.
# Key responsibilities: Maintain the isolated dashboard venv, validate imports, install the systemd unit, and restart the console.
# Operational constraints: The console is read-only and must remain isolated from the trading-engine Python environment.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
VENV_DIR="${APP_DIR}/.streamlit-venv"
SERVICE_SOURCE="${APP_DIR}/deploy/systemd/niftybot-streamlit.service"
SERVICE_TARGET="/etc/systemd/system/niftybot-streamlit.service"

cd "${APP_DIR}"

if ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --quiet -r dashboard/requirements.txt
"${VENV_DIR}/bin/python" -m py_compile dashboard/event_buffer.py dashboard/log_export.py dashboard/operations_console.py
PYTHONPATH="${APP_DIR}" "${VENV_DIR}/bin/python" -c \
  'from dashboard.event_buffer import deduplicate_events; from dashboard.log_export import window_epochs; assert callable(deduplicate_events); assert callable(window_epochs)'

sudo install -m 0644 "${SERVICE_SOURCE}" "${SERVICE_TARGET}"
sudo systemctl daemon-reload
sudo systemctl enable --quiet niftybot-streamlit.service
sudo systemctl restart niftybot-streamlit.service

for _ in $(seq 1 20); do
  if curl -fsS --max-time 2 http://127.0.0.1:8501/ >/dev/null 2>&1; then
    echo "Streamlit console ready: http://15.206.3.6:8501"
    exit 0
  fi
  sleep 1
done

sudo systemctl status niftybot-streamlit.service --no-pager -l || true
exit 1
