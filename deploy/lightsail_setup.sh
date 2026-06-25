#!/usr/bin/env bash
# File purpose: Provision and update the production AWS Lightsail host.
# Key responsibilities: Install the service, preserve credentials, validate new revisions, deploy atomically, and roll back failed releases.
# Operational constraints: Never overwrite the existing .env, never restart onto an unvalidated revision, and keep one systemd-owned application process.
set -euo pipefail

REPO="https://github.com/kundakarlabp/nifty_scalper_bot.git"
APP_DIR="/home/ubuntu/nifty_scalper_bot"
ENV_FILE="$APP_DIR/.env"
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

log "Synchronising repository"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --quiet origin main
  git -C "$APP_DIR" reset --hard --quiet origin/main
else
  git clone --quiet "$REPO" "$APP_DIR"
fi

log "Creating Python environment"
cd "$APP_DIR"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e .
python -m pip install --quiet pytest python-multipart uvicorn fastapi

if [ ! -f "$ENV_FILE" ]; then
  log "Creating .env with disabled execution defaults"
  ADMIN_PW="$(python3 -c 'import secrets;print(secrets.token_urlsafe(9))')"
  cat > "$ENV_FILE" <<EOF
# Managed locally on the Lightsail host. Do not commit or share this file.
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
EOF
  chmod 600 "$ENV_FILE"
else
  log "Preserving existing .env and credentials"
  ADMIN_PW="$(grep -E '^ADMIN_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
  ensure_env_default DEPLOYMENT_PLATFORM aws_lightsail
  ensure_env_default ALLOW_DEBUG_ENV false
fi

log "Installing systemd service"
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
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=3
TimeoutStopSec=30
KillSignal=SIGINT
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Dashboard and updater may restart only this service.
echo "ubuntu ALL=(ALL) NOPASSWD: /bin/systemctl restart ${SERVICE}, /usr/bin/systemctl restart ${SERVICE}, /bin/systemctl restart --no-block ${SERVICE}, /usr/bin/systemctl restart --no-block ${SERVICE}" | \
  sudo tee /etc/sudoers.d/niftybot >/dev/null
sudo chmod 440 /etc/sudoers.d/niftybot

sudo systemctl daemon-reload
sudo systemctl enable --quiet ${SERVICE}

log "Installing validated auto-deployer"
sudo tee /usr/local/bin/niftybot-autodeploy.sh >/dev/null <<EOF
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$APP_DIR"
SERVICE="$SERVICE"
PORT="$PORT"
STATUS_FILE="$STATUS_FILE"
VENV="$APP_DIR/.venv"

exec 9>/tmp/niftybot-autodeploy.lock
flock -n 9 || exit 0
cd "\$APP_DIR"
mkdir -p "\$(dirname "\$STATUS_FILE")"

write_status() {
  local state="\$1"
  local message="\$2"
  local escaped
  escaped="\$(printf '%s' "\$message" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  printf '{"state":"%s","message":"%s","updated_at":"%s"}\n' \
    "\$state" "\$escaped" "\$(date -Is)" > "\$STATUS_FILE.tmp"
  mv "\$STATUS_FILE.tmp" "\$STATUS_FILE"
}

BEFORE="\$(git rev-parse HEAD 2>/dev/null || echo none)"
if ! git fetch --quiet origin main; then
  write_status fetch_failed "git fetch failed"
  exit 0
fi
AFTER="\$(git rev-parse origin/main 2>/dev/null || echo none)"

if [ "\$BEFORE" = "\$AFTER" ]; then
  write_status current "running \${BEFORE:0:7}"
  exit 0
fi

write_status validating "validating \${AFTER:0:7}"
CANDIDATE="/tmp/niftybot-candidate-\${AFTER:0:12}"
rm -rf "\$CANDIDATE"
git worktree add --detach --quiet "\$CANDIDATE" "\$AFTER"
cleanup() {
  git worktree remove --force "\$CANDIDATE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if ! PYTHONPATH="\$CANDIDATE/src" "\$VENV/bin/python" -m compileall -q "\$CANDIDATE/src"; then
  write_status validation_failed "compile failed for \${AFTER:0:7}"
  exit 0
fi

TARGETED_TESTS=(
  tests/architecture/test_file_header_standard.py
  tests/architecture/test_canonical_bo_ownership.py
  tests/test_execution_path_contract.py
  tests/execution/test_runtime_order_facade.py
  tests/execution/test_runtime_bracket_facade.py
  tests/integration/test_canonical_bo_end_to_end.py
)
EXISTING_TESTS=()
for test_path in "\${TARGETED_TESTS[@]}"; do
  [ -f "\$CANDIDATE/\$test_path" ] && EXISTING_TESTS+=("\$CANDIDATE/\$test_path")
done
if [ "\${#EXISTING_TESTS[@]}" -gt 0 ]; then
  if ! PYTHONPATH="\$CANDIDATE/src" "\$VENV/bin/python" -m pytest -q "\${EXISTING_TESTS[@]}"; then
    write_status validation_failed "focused tests failed for \${AFTER:0:7}"
    exit 0
  fi
fi

write_status deploying "deploying \${AFTER:0:7}"
git reset --hard --quiet "\$AFTER"
if ! "\$VENV/bin/python" -m pip install --quiet -e .; then
  git reset --hard --quiet "\$BEFORE"
  "\$VENV/bin/python" -m pip install --quiet -e . || true
  write_status install_failed "install failed; restored \${BEFORE:0:7}"
  exit 0
fi

sudo systemctl restart "\$SERVICE"
for _ in \$(seq 1 15); do
  if curl -fsS --max-time 2 "http://127.0.0.1:\$PORT/livez" >/dev/null; then
    write_status deployed "deployed \${AFTER:0:7}"
    logger -t niftybot-autodeploy "validated and deployed \$BEFORE -> \$AFTER"
    exit 0
  fi
  sleep 2
done

git reset --hard --quiet "\$BEFORE"
"\$VENV/bin/python" -m pip install --quiet -e . || true
sudo systemctl restart "\$SERVICE"
write_status rolled_back "health check failed; restored \${BEFORE:0:7}"
logger -t niftybot-autodeploy "health check failed; restored \$AFTER -> \$BEFORE"
EOF
sudo chmod +x /usr/local/bin/niftybot-autodeploy.sh

sudo tee /etc/systemd/system/niftybot-autodeploy.service >/dev/null <<EOF
[Unit]
Description=Validate and deploy Nifty Bot from GitHub
After=network-online.target

[Service]
Type=oneshot
User=ubuntu
ExecStart=/usr/local/bin/niftybot-autodeploy.sh
EOF

sudo tee /etc/systemd/system/niftybot-autodeploy.timer >/dev/null <<EOF
[Unit]
Description=Check Nifty Bot releases every two minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
Persistent=true

[Install]
WantedBy=timers.target
EOF

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

sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
{
  auto_https disable_redirects
}
:443 {
  tls internal
  reverse_proxy 127.0.0.1:${PORT}
}
EOF
sudo systemctl restart caddy || log "Caddy restart failed; HTTP remains available on port ${PORT}"

IP="$(curl -fsSL https://checkip.amazonaws.com 2>/dev/null || echo 'YOUR_STATIC_IP')"
printf '\n============================================================\n'
printf 'Setup complete.\n'
printf 'Dashboard: https://%s/admin\n' "$IP"
printf 'Fallback:  http://%s:%s/admin\n' "$IP" "$PORT"
printf 'Password:  %s\n' "${ADMIN_PW:-already configured}"
printf 'Allow TCP 443 in Lightsail Networking. Port %s is required only for direct fallback access.\n' "$PORT"
printf 'Add %s to the Zerodha allowed IP list.\n' "$IP"
printf '============================================================\n'
