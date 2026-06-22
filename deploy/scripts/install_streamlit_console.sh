#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
VENV_DIR="${APP_DIR}/.streamlit-venv"
SERVICE_SOURCE="${APP_DIR}/deploy/systemd/niftybot-streamlit.service"
SERVICE_TARGET="/etc/systemd/system/niftybot-streamlit.service"

cd "${APP_DIR}"

if ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3-venv
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r dashboard/requirements.txt

sudo install -m 0644 "${SERVICE_SOURCE}" "${SERVICE_TARGET}"
sudo systemctl daemon-reload
sudo systemctl enable --now niftybot-streamlit.service
sudo systemctl restart niftybot-streamlit.service
sudo systemctl status niftybot-streamlit.service --no-pager

echo "Console URL: http://15.206.3.6:8501"
