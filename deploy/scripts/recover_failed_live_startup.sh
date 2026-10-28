#!/usr/bin/env bash
# File purpose: Recover a failed LIVE startup through a CI-proven release.
# Key responsibilities: Verify broker safety and run the validated candidate guard.
# Operational constraints: No direct restart, unverified token or risk override.
set -euo pipefail

APP_DIR="/home/ubuntu/nifty_scalper_bot"
CONFIG_DIR="/home/ubuntu/.config/niftybot"
ENV_FILE="$CONFIG_DIR/niftybot.env"
PROOF="$CONFIG_DIR/auto_release_verified_token_rotation"
MARKER="$CONFIG_DIR/auto_release_restart_pending"
REMOTE="refs/tags/nifty-ci-passed/"

die() { printf '[recovery] BLOCKED: %s\n' "$*" >&2; exit 1; }
[ "$(id -un)" = "ubuntu" ] || die "execute as ubuntu, not root"
[ -f "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ] || die "protected environment missing or symlinked"
[ -x "$APP_DIR/.venv/bin/python" ] || die "production Python unavailable"
cd "$APP_DIR"
[ -d .git ] || die "production Git checkout missing"
git remote get-url origin | grep -Eq 'github\.com[:/]kundakarlabp/nifty_scalper_bot(\.git)?$' ||
  die "unexpected Git origin"
git fetch --quiet origin main || die "GitHub fetch failed"
target="$(git rev-parse refs/remotes/origin/main)"
[[ "$target" =~ ^[0-9a-f]{40}$ ]] || die "invalid main commit"
ref="$REMOTE$target"
proof_ref="$(git ls-remote --refs origin "$ref")" ||
  die "post-merge CI proof could not be fetched"
[ "$proof_ref" = "$(printf '%s\t%s' "$target" "$ref")" ] ||
  die "exact main SHA has no successful CI proof"
echo "[recovery] Exact-main CI proof verified; checking saved broker session"

# The old installed CLI did not publish token-rotation proof. Independently
# verify the saved session and broker-flat state without changing credentials.
export BOT_ENV_FILE="$ENV_FILE" NIFTYBOT_CONFIG_DIR="$CONFIG_DIR"
"$APP_DIR/.venv/bin/python" - <<'PY_RECOVER'
import hashlib
import hmac
import json
import math
import os
import stat
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values
from kiteconnect import KiteConnect

config = Path(os.environ["NIFTYBOT_CONFIG_DIR"])
env_file = Path(os.environ["BOT_ENV_FILE"])
env = dotenv_values(env_file)
def first(*names):
    return next((str(env[n]).strip() for n in names if env.get(n)), "")
def matching(*names):
    values = {str(env[n]).strip() for n in names if env.get(n)}
    if len(values) != 1:
        raise RuntimeError("missing_or_conflicting_" + names[0])
    return values.pop()
def probe(path):
    try:
        response = urllib.request.urlopen("http://127.0.0.1:8080" + path, timeout=5)
    except urllib.error.HTTPError as exc:
        if exc.code != 503:
            raise
        response = exc
    with response:
        return json.load(response)

assert env.get("ENABLE_LIVE", "").lower() == "true", "not_configured_live"
assert env.get("EXECUTION_MODE", "").upper() == "LIVE", "not_execution_live"
key = matching("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
token = matching("BROKER_ACCESS_TOKEN", "ZERODHA_ACCESS_TOKEN", "KITE_ACCESS_TOKEN")
owner = first("ADMIN_ZERODHA_USER_ID", "ZERODHA_USER_ID", "KITE_USER_ID")
assert owner, "expected_owner_not_configured"
segment = first("BROKER_MARGIN_SEGMENT") or "equity"
assert segment in {"equity", "commodity"}, "invalid_margin_segment"
live = probe("/livez")
ready = probe("/readyz")
health = probe("/health/trading")
release = probe("/releasez")
assert live.get("engine_http_responsive") is True, "engine_not_responsive"
assert live.get("quiet") is not True and live.get("bot_loaded") is False, "engine_not_unloaded"
assert ready.get("status") == "degraded", "unexpected_readiness"
assert ready.get("primary_blocker") == "startup_failed", "not_startup_failed"
assert ready.get("error") == "LIVE startup requires validated broker balance", "unexpected_startup_error"
assert health.get("status") == "starting", "unexpected_trading_status"
assert health.get("primary_blocker") == "startup_incomplete", "unexpected_trading_blocker"
assert health.get("live_orders_armed") is False, "entry_path_not_disarmed"
assert str(release.get("effective_sha") or "").startswith(
    os.popen("git rev-parse HEAD").read().strip()[:12]
), "running_build_mismatch"

kite = KiteConnect(api_key=key, timeout=8)
kite.set_access_token(token)
profile = kite.profile()
assert hmac.compare_digest(str(profile.get("user_id") or ""), owner), "broker_owner_mismatch"
funds = kite.margins(segment)
available = funds.get("available") if isinstance(funds, dict) else None
assert isinstance(available, dict), "margin_payload_missing"
cash = available.get("live_balance")
if cash is None:
    cash = available.get("cash")
assert cash is not None and math.isfinite(float(cash)) and float(cash) >= 0, "funds_invalid"

terminals = {"COMPLETE", "CANCELLED", "REJECTED", "LAPSED"}
for _ in range(2):
    positions = kite.positions()
    orders = kite.orders()
    assert isinstance(positions, dict) and all(
        isinstance(positions.get(k), list) for k in ("net", "day")
    ), "positions_shape_invalid"
    assert isinstance(orders, list), "orders_shape_invalid"
    assert all(
        isinstance(row, dict) and "quantity" in row and float(row["quantity"]) == 0
        for row in positions["net"] + positions["day"]
    ), "broker_position_nonflat"
    assert all(
        isinstance(row, dict) and str(row.get("status") or "").upper() in terminals
        for row in orders
    ), "broker_order_not_terminal"
    time.sleep(1)

assert config.is_dir() and not config.is_symlink(), "unsafe_config_directory"
proof = config / "auto_release_verified_token_rotation"
marker = config / "auto_release_restart_pending"
assert not proof.is_symlink() and not marker.is_symlink(), "symlinked_recovery_marker"
fd, temporary = tempfile.mkstemp(prefix=".verified-token.", dir=config)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as target:
        json.dump(
            {"token_sha256": hashlib.sha256(token.encode()).hexdigest(),
             "verified_at": time.time()},
            target,
        )
        target.flush()
        os.fsync(target.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, proof)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
try:
    assert stat.S_ISREG(os.fstat(descriptor).st_mode), "unsafe_restart_marker"
    os.fchmod(descriptor, 0o600)
finally:
    os.close(descriptor)
print("[recovery] Owner, funds, broker-flat snapshots and rotation proof verified")
PY_RECOVER

# Execute the CI-verified candidate release runner WITHOUT modifying the
# deployed checkout first. This repairs the old runner's recovery deadlock.
release_file="$(mktemp /tmp/nifty-recovery-release.XXXXXX)"
trap 'rm -f "$release_file"' EXIT
git show "$target:deploy/lightsail_release.sh" > "$release_file" ||
  die "failed to retrieve verified release runner"
bash -n "$release_file" || die "candidate release runner syntax invalid"
echo "[recovery] Invoking validated release guard; no direct systemctl restart"
bash "$release_file" --auto --wait-for-lock || die "validated release runner failed"
current="$(git rev-parse HEAD)"
printf '[recovery] Deployed checkout: %.12s; expected: %.12s\n' "$current" "$target"
if [ "$current" != "$target" ]; then
  cat "$APP_DIR/data/auto_update_status.json" 2>/dev/null || true
  die "release deferred or failed; checkout was not changed"
fi
for _ in $(seq 1 12); do
  ready="$("$APP_DIR/.venv/bin/python" - <<'PY_READY'
import json, urllib.error, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8080/readyz", timeout=3) as r:
        print("true" if json.load(r).get("ready") is True else "false")
except (urllib.error.URLError, ValueError, TimeoutError):
    print("false")
PY_READY
)"
  [ "$ready" = "true" ] && { echo "[recovery] VERIFIED: updated engine ready"; exit 0; }
  sleep 3
done
cat "$APP_DIR/data/auto_update_status.json" 2>/dev/null || true
die "new checkout installed but engine readiness is not verified"
