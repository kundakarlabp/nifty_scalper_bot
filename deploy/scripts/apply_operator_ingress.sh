#!/usr/bin/env bash
# File purpose: Synchronize Lightsail admin bind and public TLS configuration.
# Key responsibilities: Validate Caddy, install loopback admin unit, reload and restore on failure.
# Operational constraints: Do not restart trading or change credentials; never claim certificate issuance from reload.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
ADMIN_SOURCE="$APP_DIR/deploy/systemd/niftybot-admin.service"
ADMIN_TARGET="/etc/systemd/system/niftybot-admin.service"
CADDY_SOURCE="$APP_DIR/deploy/caddy/Caddyfile"
CADDY_TARGET="/etc/caddy/Caddyfile"

# The independent operator API must bind to loopback even if HTTPS
# configuration cannot be validated. Never restart the trading engine.
if ! cmp -s "$ADMIN_SOURCE" "$ADMIN_TARGET"; then
  sudo -n install -m 0644 "$ADMIN_SOURCE" "$ADMIN_TARGET"
  sudo -n systemctl daemon-reload
  sudo -n systemctl restart niftybot-admin.service
fi

caddy_supported() {
  command -v caddy >/dev/null 2>&1 || return 1
  local version
  version="$(caddy version | awk '{print $1}' | sed 's/^v//')"
  [ -n "$version" ] && dpkg --compare-versions "$version" ge "2.10.1"
}

# Use only the already configured free stable APT package, and never install
# or upgrade it during active trading. Limit attempts to once per six hours.
if ! caddy_supported; then
  marker="${NIFTYBOT_CONFIG_DIR:-/home/ubuntu/.config/niftybot}/caddy_upgrade_attempted"
  if [ ! -f "$marker" ] || [ -n "$(find "$marker" -mmin +360 -print 2>/dev/null)" ]; then
    if curl -fsS --max-time 3 http://127.0.0.1:8080/livez \
      | grep -Eq '"quiet"[[:space:]]*:[[:space:]]*true'; then
      mkdir -p "$(dirname "$marker")"
      : > "$marker"
      if command -v caddy >/dev/null 2>&1; then
        sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq \
          && sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install \
            --only-upgrade -y -qq caddy || true
      fi
    fi
  fi
fi
if ! caddy_supported; then
  echo "[operator-ingress] Caddy >=2.10.1 required for IP TLS; admin is loopback-only" >&2
  exit 1
fi
if ! sudo -n caddy validate --config "$CADDY_SOURCE" --adapter caddyfile >/dev/null; then
  echo "[operator-ingress] Caddy config invalid; admin remains loopback-only" >&2
  exit 1
fi

if ! sudo -n cmp -s "$CADDY_SOURCE" "$CADDY_TARGET"; then
  previous="$(mktemp)"
  trap 'rm -f "$previous"' EXIT
  if sudo -n test -f "$CADDY_TARGET"; then
    sudo -n cp "$CADDY_TARGET" "$previous"
  else
    : > "$previous"
  fi
  sudo -n install -m 0644 "$CADDY_SOURCE" "$CADDY_TARGET"
  if ! sudo -n systemctl reload caddy; then
    echo "[operator-ingress] Caddy reload failed; restoring prior configuration" >&2
    if [ -s "$previous" ]; then
      sudo -n install -m 0644 "$previous" "$CADDY_TARGET"
      sudo -n systemctl reload caddy || true
    else
      sudo -n rm -f "$CADDY_TARGET"
    fi
    exit 1
  fi
fi

# Certificate issuance may be asynchronous: external verified-TLS checks
# are still required. Never claim that reload alone proves public HTTPS.
echo "[operator-ingress] systemd admin bind and Caddy configuration synchronized"
