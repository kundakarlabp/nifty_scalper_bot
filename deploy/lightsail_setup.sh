#!/usr/bin/env bash
# Nifty Scalper Bot — one-time Lightsail/Ubuntu setup.
# Safe to re-run: updates code and restarts in place.
set -euo pipefail

REPO="https://github.com/kundakarlabp/nifty_scalper_bot.git"
APP_DIR="/home/ubuntu/nifty_scalper_bot"
ENV_FILE="$APP_DIR/.env"
SERVICE="niftybot"
PORT="8080"

echo "==> Installing system packages (Python, git)..."
sudo apt-get update -y -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git

echo "==> Fetching the bot code..."
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --quiet origin main
  git -C "$APP_DIR" reset --hard --quiet origin/main
else
  git clone --quiet "$REPO" "$APP_DIR"
fi

echo "==> Creating Python environment and installing dependencies..."
cd "$APP_DIR"
python3 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet python-multipart uvicorn fastapi

# --- .env: create on first run, preserve on re-run ---
if [ ! -f "$ENV_FILE" ]; then
  echo "==> Creating .env with safe defaults (you fill credentials in the dashboard)..."
  ADMIN_PW="$(python3 -c 'import secrets;print(secrets.token_urlsafe(9))')"
  cat > "$ENV_FILE" <<EOF
# Filled in via the web dashboard. Do not share this file.
ADMIN_PASSWORD=$ADMIN_PW
PORT=$PORT
BOT_ENV_FILE=$ENV_FILE
BOT_SERVICE_NAME=$SERVICE
# Trading off until you enter credentials and turn it on in the dashboard:
ENABLE_LIVE=false
EXECUTION_MODE=SHADOW
# Zerodha (enter in dashboard):
KITE_API_KEY=
KITE_API_SECRET=
KITE_ACCESS_TOKEN=
# Telegram (optional, enter in dashboard):
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_ALLOWED_ID=
EOF
  chmod 600 "$ENV_FILE"
else
  echo "==> Existing .env kept (credentials preserved)."
  ADMIN_PW="$(grep -E '^ADMIN_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
fi

# --- systemd service: auto-start, auto-restart, survives reboot ---
echo "==> Installing background service..."
sudo tee /etc/systemd/system/${SERVICE}.service >/dev/null <<EOF
[Unit]
Description=Nifty Scalper Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Allow the dashboard's restart button to run systemctl without a password.
echo "ubuntu ALL=(ALL) NOPASSWD: /bin/systemctl restart ${SERVICE}, /usr/bin/systemctl restart ${SERVICE}" | \
  sudo tee /etc/sudoers.d/niftybot >/dev/null
sudo chmod 440 /etc/sudoers.d/niftybot

sudo systemctl daemon-reload
sudo systemctl enable --quiet ${SERVICE}
sudo systemctl restart ${SERVICE}

# --- HTTPS via Caddy reverse proxy (encrypts password/token in transit) ---
# Caddy serves HTTPS on 443 with a self-signed local certificate and forwards
# to the bot on 127.0.0.1:$PORT. Your browser will show a one-time "not trusted"
# warning (expected for an IP-only self-signed cert) — click Advanced -> Proceed.
echo "==> Installing Caddy for HTTPS..."
if ! command -v caddy >/dev/null 2>&1; then
  sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y -qq
  sudo apt-get install -y -qq caddy
fi

PUB_IP="$(curl -fsSL https://checkip.amazonaws.com 2>/dev/null || echo '')"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
{
  auto_https disable_redirects
}
:443 {
  tls internal
  reverse_proxy 127.0.0.1:${PORT}
}
EOF
sudo systemctl restart caddy || echo "   (Caddy restart issue — HTTP on ${PORT} still works)"

IP="$(curl -fsSL https://checkip.amazonaws.com 2>/dev/null || echo 'YOUR_STATIC_IP')"
echo ""
echo "============================================================"
echo " ✅ Setup complete."
echo ""
echo " Secure dashboard (recommended):  https://${IP}/admin"
echo "   (one-time browser warning -> Advanced -> Proceed; it is your own server)"
echo " Plain dashboard (fallback):       http://${IP}:${PORT}/admin"
echo " Password:                         ${ADMIN_PW}"
echo ""
echo " Firewall: allow TCP 443 (for HTTPS) and TCP ${PORT} in Lightsail Networking."
echo " Next: open the dashboard, enter Zerodha + Telegram details, Save,"
echo " check logs look healthy in SHADOW, then use the Live Trading toggle."
echo " Reminder: add ${IP} to Zerodha Allowed IPs (developers.kite.trade)."
echo "============================================================"
