#!/usr/bin/env bash
# File purpose: Provision and update the production AWS Lightsail host.
# Key responsibilities: Install the service, migrate secrets outside Git, configure validated releases, and keep one systemd-owned process.
# Operational constraints: Never store credentials in the repository; never restart onto an unvalidated revision; AWS Lightsail is the production authority.
set -euo pipefail

REPO="https://github.com/kundakarlabp/nifty_scalper_bot.git"
APP_DIR="/home/ubuntu/nifty_scalper_bot"
CONFIG_DIR="/home/ubuntu/.config/niftybot"
ENV_FILE="$CONFIG_DIR/niftybot.env"
LEGACY_ENV="$APP_DIR/.env"
SERVICE="niftybot"
PORT="8080"
STATUS_FILE="$APP_DIR/data/auto_update_status.json"

log() {
  printf '[lightsail-setup] %s\n' "$*"
}

ensure_env_default() {
  local key="$1"
  local value="$2"
  if ! grep -qE "^${key}=" "$ENV_FILE"; then
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

log "Installing system packages"
sudo apt-get update -y -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git curl util-linux

log "Preparing repository"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --quiet origin main
else
  git clone --quiet "$REPO" "$APP_DIR"
fi

log "Preparing external configuration"
mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"
if [ ! -f "$ENV_FILE" ]; then
  if [ -f "$LEGACY_ENV" ]; then
    cp -p "$LEGACY_ENV" "$ENV_FILE"
    log "Migrated existing .env outside the Git checkout"
  else
    ADMIN_PW="$(python3 -c 'import secrets;print(secrets.token_urlsafe(18))')"
    cat > "$ENV_FILE" <<EOF_ENV
# Managed locally on the Lightsail host. Never commit or share this file.
ADMIN_PASSWORD=$ADMIN_PW
PORT=$PORT
BOT_ENV_FILE=$ENV_FILE
BOT_SERVICE_NAME=$SERVICE
BOT_APP_DIR=$APP_DIR
DEPLOYMENT_PLATFORM=aws_lightsail
ENABLE_LIVE=false
EXECUTION_MODE=SHADOW
ALLOW_DEBUG_ENV=false
KITE_API_KEY=
KITE_API_SECRET=
KITE_ACCESS_TOKEN=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_ALLOWED_ID=
EOF_ENV
    log "Created an execution-disabled environment file"
  fi
else
  log "Preserving external environment and credentials"
  ADMIN_PW="$(grep -E '^ADMIN_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
fi
chmod 600 "$ENV_FILE"
ensure_env_default BOT_ENV_FILE "$ENV_FILE"
ensure_env_default BOT_SERVICE_NAME "$SERVICE"
ensure_env_default BOT_APP_DIR "$APP_DIR"
ensure_env_default DEPLOYMENT_PLATFORM aws_lightsail
ensure_env_default ALLOW_DEBUG_ENV false
ensure_env_default PORT "$PORT"

# Compatibility path for tools that still look for APP_DIR/.env. The target is
# outside Git, so reset/pull operations cannot overwrite credentials.
if [ -e "$LEGACY_ENV" ] && [ ! -L "$LEGACY_ENV" ]; then
  mv "$LEGACY_ENV" "$CONFIG_DIR/legacy.env.$(date +%Y%m%d_%H%M%S)"
fi
ln -sfn "$ENV_FILE" "$LEGACY_ENV"

log "Creating Python environment"
cd "$APP_DIR"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e .
python -m pip install --quiet pytest python-multipart uvicorn fastapi

log "Installing systemd service"
sudo tee /etc/systemd/system/${SERVICE}.service >/dev/null <<EOF_UNIT
[Unit]
Description=Nifty Scalper Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
Environment=BOT_ENV_FILE=$ENV_FILE
Environment=BOT_SERVICE_NAME=$SERVICE
Environment=BOT_APP_DIR=$APP_DIR
Environment=DEPLOYMENT_PLATFORM=aws_lightsail
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=3
TimeoutStopSec=30
KillSignal=SIGINT
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF_UNIT

# Dashboard and release script may restart only the trading service.
echo "ubuntu ALL=(ALL) NOPASSWD: /bin/systemctl restart ${SERVICE}, /usr/bin/systemctl restart ${SERVICE}, /bin/systemctl restart --no-block ${SERVICE}, /usr/bin/systemctl restart --no-block ${SERVICE}" | \
  sudo tee /etc/sudoers.d/niftybot >/dev/null
sudo chmod 440 /etc/sudoers.d/niftybot

sudo systemctl daemon-reload
sudo systemctl enable --quiet ${SERVICE}

log "Installing validated auto-deployer"
chmod +x "$APP_DIR/deploy/lightsail_release.sh"
sudo tee /etc/systemd/system/niftybot-autodeploy.service >/dev/null <<EOF_DEPLOY_SERVICE
[Unit]
Description=Validate and deploy Nifty Bot from GitHub
After=network-online.target

[Service]
Type=oneshot
User=ubuntu
Environment=BOT_APP_DIR=$APP_DIR
Environment=BOT_ENV_FILE=$ENV_FILE
Environment=BOT_SERVICE_NAME=$SERVICE
Environment=PORT=$PORT
ExecStart=$APP_DIR/deploy/lightsail_release.sh --auto
EOF_DEPLOY_SERVICE

sudo tee /etc/systemd/system/niftybot-autodeploy.timer >/dev/null <<EOF_TIMER
[Unit]
Description=Check Nifty Bot releases every two minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
Persistent=true

[Install]
WantedBy=timers.target
EOF_TIMER

sudo systemctl daemon-reload
sudo systemctl enable --quiet --now niftybot-autodeploy.timer
sudo systemctl restart ${SERVICE}

log "Installing Caddy HTTPS reverse proxy"
if ! command -v caddy >/dev/null 2>&1; then
  sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y -qq
  sudo apt-get install -y -qq caddy
fi

sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF_CADDY
{
  auto_https disable_redirects
}
:443 {
  tls internal
  reverse_proxy 127.0.0.1:${PORT}
}
EOF_CADDY
sudo systemctl restart caddy || log "Caddy restart failed; HTTP remains available on port ${PORT}"

IP="$(curl -fsSL https://checkip.amazonaws.com 2>/dev/null || echo 'YOUR_STATIC_IP')"
printf '\n============================================================\n'
printf 'Setup complete.\n'
printf 'Dashboard: https://%s/admin\n' "$IP"
printf 'Fallback:  http://%s:%s/admin\n' "$IP" "$PORT"
printf 'Environment: %s\n' "$ENV_FILE"
printf 'Password:  %s\n' "${ADMIN_PW:-already configured}"
printf 'Allow TCP 443 in Lightsail Networking. Port %s is required only for direct fallback access.\n' "$PORT"
printf 'Add %s to the Zerodha allowed IP list.\n' "$IP"
printf '============================================================\n'
