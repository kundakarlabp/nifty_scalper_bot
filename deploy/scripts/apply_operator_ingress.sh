#!/usr/bin/env bash
# Purpose: One guarded, idempotent Lightsail operator-ingress hardening step.
# Boundaries: Never restart the trading engine or alter private broker settings.
# Safety: Preserve the previous Caddyfile when validation/reload fails.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
ADMIN_SOURCE="$APP_DIR/deploy/systemd/niftybot-admin.service"
ADMIN_TARGET="/etc/systemd/system/niftybot-admin.service"
CADDY_SOURCE="$APP_DIR/deploy/caddy/Caddyfile"
CADDY_TARGET="/etc/caddy/Caddyfile"

if ! command -v caddy >/dev/null 2>&1; then
  echo "[operator-ingress] Caddy is missing; HTTPS configuration not applied" >&2
  exit 1
fi
caddy_version="$(caddy version | sed -nE 's/^v([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' | head -1)"
if [ -z "$caddy_version" ] || ! dpkg --compare-versions "$caddy_version" ge "2.10.1"; then
  echo "[operator-ingress] Caddy >=2.10.1 needed for ACME IP certificate" >&2
  exit 1
fi
if ! sudo -n caddy validate --config "$CADDY_SOURCE" --adapter caddyfile >/dev/null; then
  echo "[operator-ingress] Candidate Caddyfile validation failed" >&2
  exit 1
fi

# Independent admin API is reachable from the host and Caddy only.
# This change does not stop or restart the trading engine.
if ! cmp -s "$ADMIN_SOURCE" "$ADMIN_TARGET"; then
  sudo -n install -m 0644 "$ADMIN_SOURCE" "$ADMIN_TARGET"
  sudo -n systemctl daemon-reload
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
