#!/usr/bin/env bash
# File purpose: Validate and deploy the production AWS Lightsail release.
# Key responsibilities: Preserve host secrets, validate candidate code, restart systemd services, verify readiness, and roll back failed bot revisions.
# Operational constraints: Secrets live outside the Git checkout; only one deployment runs at a time; Streamlit failure must not interrupt a healthy trading engine.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
SERVICE="${BOT_SERVICE_NAME:-niftybot}"
STREAMLIT_SERVICE="${BOT_STREAMLIT_SERVICE_NAME:-niftybot-streamlit}"
ADMIN_SERVICE="${BOT_ADMIN_SERVICE_NAME:-niftybot-admin}"
AUTODEPLOY_SERVICE="${BOT_AUTODEPLOY_SERVICE_NAME:-niftybot-autodeploy}"
PORT="${PORT:-8080}"
ADMIN_PORT="${BOT_ADMIN_PORT:-8081}"
STREAMLIT_PORT="${BOT_STREAMLIT_PORT:-8501}"
CONFIG_DIR="${NIFTYBOT_CONFIG_DIR:-/home/ubuntu/.config/niftybot}"
ENV_FILE="${BOT_ENV_FILE:-$CONFIG_DIR/niftybot.env}"
STATUS_FILE="${BOT_UPDATE_STATUS_FILE:-$APP_DIR/data/auto_update_status.json}"
VENV="${BOT_VENV:-$APP_DIR/.venv}"
STREAMLIT_VENV="${BOT_STREAMLIT_VENV:-$APP_DIR/.streamlit-venv}"
LOCK_FILE="${BOT_DEPLOY_LOCK_FILE:-/tmp/niftybot-deploy.lock}"
RESTART_PENDING_FILE="${CONFIG_DIR}/auto_release_restart_pending"
VERIFIED_TOKEN_ROTATION_FILE="${CONFIG_DIR}/auto_release_verified_token_rotation"
RELEASE_HOLD_FILE="${CONFIG_DIR}/auto_release_entry_hold"
RELEASE_ENTRY_LOCK_FILE="${CONFIG_DIR}/auto_release_entry_lock"
RELEASE_HOLD_ACTIVE=false
FORCE_RESTART=false
AUTO_MODE=false
SYSTEMD_ENTRYPOINT_MIGRATED=false
AUTODEPLOY_ENTRYPOINT_MIGRATED=false
RUNTIME_ENV_CHANGED=false
PACKAGE_INSTALL_ATTEMPTED=false
STREAMLIT_DEPENDENCY_FAILED=false

for arg in "$@"; do
  case "$arg" in
    --force|--force-restart) FORCE_RESTART=true ;;
    --auto) AUTO_MODE=true ;;
    *) printf '[lightsail-release] unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

log() { printf '[lightsail-release] %s\n' "$*"; }

write_status() {
  local state="$1" message="$2" escaped
  mkdir -p "$(dirname "$STATUS_FILE")"
  escaped="$(printf '%s' "$message" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  printf '{"state":"%s","message":"%s","updated_at":"%s"}\n' \
    "$state" "$escaped" "$(date -Is)" > "$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

first_nonempty_env() {
  local key value
  for key in "$@"; do
    value="$(grep -E "^${key}=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    if [ -n "${value//[[:space:]]/}" ]; then printf '%s' "$value"; return 0; fi
  done
  return 1
}

env_truthy() {
  local value
  value="$(first_nonempty_env "$1" 2>/dev/null || true)"
  case "${value,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac
}

ensure_env_default() {
  local key="$1" value="$2"
  if ! grep -qE "^${key}=" "$ENV_FILE" 2>/dev/null; then
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    chmod 600 "$ENV_FILE" 2>/dev/null || true
    RUNTIME_ENV_CHANGED=true
  fi
}

set_runtime_build_sha() {
  local sha="$1" tmp
  if [[ ! "$sha" =~ ^[0-9a-fA-F]{7,64}$ ]]; then
    log "ERROR: refusing invalid runtime build SHA: $sha"
    return 1
  fi
  tmp="$(mktemp "${ENV_FILE}.build-sha.XXXXXX")" || return 1
  if ! awk -v sha="$sha" '
    BEGIN { written = 0 }
    /^GIT_COMMIT_SHA=/ {
      if (!written) { print "GIT_COMMIT_SHA=" sha; written = 1 }
      next
    }
    { print }
    END { if (!written) print "GIT_COMMIT_SHA=" sha }
  ' "$ENV_FILE" > "$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  chmod 600 "$tmp" || { rm -f "$tmp"; return 1; }
  mv "$tmp" "$ENV_FILE"
}

validate_environment() {
  if [ ! -f "$ENV_FILE" ]; then
    write_status env_missing "environment file missing: $ENV_FILE"
    log "ERROR: environment file missing: $ENV_FILE"
    return 1
  fi
  chmod 600 "$ENV_FILE" 2>/dev/null || true
  if ! first_nonempty_env BROKER_API_KEY ZERODHA_API_KEY KITE_API_KEY >/dev/null; then
    write_status env_invalid "broker API key is missing"; return 1
  fi
  if ! first_nonempty_env BROKER_API_SECRET ZERODHA_API_SECRET KITE_API_SECRET >/dev/null; then
    write_status env_invalid "broker API secret is missing"; return 1
  fi
  if env_truthy ENABLE_LIVE; then
    local execution_mode
    execution_mode="$(first_nonempty_env EXECUTION_MODE 2>/dev/null || true)"
    if [ "${execution_mode^^}" != "LIVE" ]; then
      write_status env_invalid "ENABLE_LIVE=true requires EXECUTION_MODE=LIVE"; return 1
    fi
    if ! first_nonempty_env BROKER_ACCESS_TOKEN ZERODHA_ACCESS_TOKEN KITE_ACCESS_TOKEN >/dev/null; then
      write_status env_invalid "LIVE mode requires a broker access token"; return 1
    fi
  fi
}

broker_auth_dependency_degraded() {
  local ready_json
  ready_json="$(curl -sS --max-time 3 "http://127.0.0.1:${PORT}/readyz" 2>/dev/null || true)"
  grep -Eq '"primary_blocker"[[:space:]]*:[[:space:]]*"startup_failed"' <<<"$ready_json" || return 1
  grep -Eqi '"error"[[:space:]]*:[[:space:]]*"[^"]*(zerodha authentication invalid|incorrect[^"]*api_key[^"]*access_token|invalid access_token|session expired|token expired|authentication failed)' <<<"$ready_json"
}

service_healthy() {
  local expected_sha="${1:-}" live_json release_json expected_short
  live_json="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/livez" 2>/dev/null || true)"
  grep -Eq '"engine_http_responsive"[[:space:]]*:[[:space:]]*true' <<<"$live_json" || return 1

  if [ -n "$expected_sha" ]; then
    expected_short="${expected_sha:0:12}"
    release_json="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/releasez" 2>/dev/null || true)"
    grep -Eq "\"effective_sha\"[[:space:]]*:[[:space:]]*\"${expected_short}\"" <<<"$release_json" || return 1
  fi

  if grep -Eq '"bot_loaded"[[:space:]]*:[[:space:]]*true' <<<"$live_json"; then
    return 0
  fi
  # deployment_main intentionally unloads the trading stack outside the AUTO
  # window. That explicit quiet control plane is healthy even though
  # bot_loaded=false.
  if grep -Eq '"quiet"[[:space:]]*:[[:space:]]*true' <<<"$live_json" \
    && grep -Eq '"bot_loaded"[[:space:]]*:[[:space:]]*false' <<<"$live_json"; then
    return 0
  fi
  # A broker-token/session failure is an external dependency failure, not proof
  # that the candidate code release is unhealthy. Accept only this explicit
  # fail-closed startup shape; unknown startup failures still trigger rollback.
  broker_auth_dependency_degraded
}

service_healthy_confirmed() {
  local expected_sha="${1:-}"
  local attempts="${BOT_DEPLOY_HEALTH_PROBE_ATTEMPTS:-3}"
  local delay="${BOT_DEPLOY_HEALTH_PROBE_INTERVAL_SEC:-2}"
  local attempt

  if ! [[ "$attempts" =~ ^[1-9][0-9]*$ ]]; then attempts=3; fi
  if ! [[ "$delay" =~ ^[0-9]+([.][0-9]+)?$ ]]; then delay=2; fi

  for attempt in $(seq 1 "$attempts"); do
    if service_healthy "$expected_sha"; then
      if [ "$attempt" -gt 1 ]; then
        log "health probe recovered on attempt $attempt/$attempts; restart suppressed"
      fi
      return 0
    fi
    if [ "$attempt" -lt "$attempts" ]; then
      sleep "$delay"
    fi
  done
  log "health probe failed $attempts consecutive attempts"
  return 1
}

wait_for_service() {
  local expected_sha="${1:-}"
  for _ in $(seq 1 150); do service_healthy "$expected_sha" && return 0; sleep 2; done
  return 1
}

migrate_systemd_entrypoint() {
  local unit_path="/etc/systemd/system/${SERVICE}.service"
  local canonical_exec="ExecStart=${VENV}/bin/python -m uvicorn nifty_scalper_bot.deployment_main:app --host 0.0.0.0 --port ${PORT}"
  if [ ! -f "$unit_path" ]; then
    SYSTEMD_ENTRYPOINT_MIGRATED=false
    return 0
  fi
  if grep -Fqx "$canonical_exec" "$unit_path" 2>/dev/null; then
    SYSTEMD_ENTRYPOINT_MIGRATED=false
    return 0
  fi
  if ! grep -q '^ExecStart=' "$unit_path" 2>/dev/null; then
    log "WARNING: $unit_path has no ExecStart; skipping entrypoint migration"
    SYSTEMD_ENTRYPOINT_MIGRATED=false
    return 0
  fi
  sudo python3 - "$unit_path" "$canonical_exec" <<'PY_MIGRATE'
import sys
from pathlib import Path
unit = Path(sys.argv[1])
canonical = sys.argv[2]
text = unit.read_text(encoding="utf-8")
lines = text.splitlines()
out = []
changed = False
for line in lines:
    if line.startswith("ExecStart=") and not changed:
        out.append(canonical)
        changed = True
    else:
        out.append(line)
if changed:
    unit.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PY_MIGRATE
  sudo systemctl daemon-reload
  SYSTEMD_ENTRYPOINT_MIGRATED=true
  log "migrated $SERVICE ExecStart to deployment_main:app; env file preserved"
}

migrate_autodeploy_entrypoint() {
  local unit_path="/etc/systemd/system/${AUTODEPLOY_SERVICE}.service"
  local canonical_exec="ExecStart=/usr/bin/env bash ${APP_DIR}/deploy/lightsail_release.sh --auto"
  if [ ! -f "$unit_path" ]; then
    AUTODEPLOY_ENTRYPOINT_MIGRATED=false
    return 0
  fi
  if grep -Fqx "$canonical_exec" "$unit_path" 2>/dev/null; then
    AUTODEPLOY_ENTRYPOINT_MIGRATED=false
    return 0
  fi
  if ! grep -q '^ExecStart=' "$unit_path" 2>/dev/null; then
    log "WARNING: $unit_path has no ExecStart; skipping auto-deploy entrypoint migration"
    AUTODEPLOY_ENTRYPOINT_MIGRATED=false
    return 0
  fi
  sudo python3 - "$unit_path" "$canonical_exec" <<'PY_MIGRATE'
import sys
from pathlib import Path
unit = Path(sys.argv[1])
canonical = sys.argv[2]
text = unit.read_text(encoding="utf-8")
lines = text.splitlines()
out = []
changed = False
for line in lines:
    if line.startswith("ExecStart=") and not changed:
        out.append(canonical)
        changed = True
    else:
        out.append(line)
if changed:
    unit.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PY_MIGRATE
  sudo systemctl daemon-reload
  AUTODEPLOY_ENTRYPOINT_MIGRATED=true
  log "migrated $AUTODEPLOY_SERVICE ExecStart to bash wrapper"
}

configure_autodeploy_timer() {
  # Upgrade the existing host timer in place, without running the full installer
  # or stopping the engine. Run at most once per host (idempotent).
  local timer="/etc/systemd/system/${AUTODEPLOY_SERVICE}.timer"
  [ -f "$timer" ] || return 0
  if grep -Fqx "OnUnitInactiveSec=30s" "$timer" 2>/dev/null; then
    return 0
  fi
  if sudo -n python3 - "$timer" <<'PY_TIMER'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
out = []
for line in lines:
    if line.startswith("OnBootSec="):
        out.append("OnBootSec=30s")
    elif line.startswith("OnUnitActiveSec=") or line.startswith("OnUnitInactiveSec="):
        continue
    elif line.startswith("AccuracySec="):
        continue
    else:
        out.append(line)
idx = out.index("[Timer]") + 1
out[idx:idx] = ["OnUnitInactiveSec=30s", "AccuracySec=5s"]
path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PY_TIMER
  then
    sudo -n systemctl daemon-reload &&
      sudo -n systemctl restart "${AUTODEPLOY_SERVICE}.timer" &&
      log "auto-deploy timer upgraded to 30 seconds after each completed run" ||
      log "WARNING: timer reload failed; existing timer continues"
  else
    log "WARNING: timer upgrade needs host permissions; existing timer continues"
  fi
}

admin_healthy() {
  # Verify the independent control plane, not just the trading engine.
  curl -fsS --max-time 3 "http://127.0.0.1:${ADMIN_PORT}/healthz" >/dev/null 2>&1 &&
    curl -fsS --max-time 4 "http://127.0.0.1:${ADMIN_PORT}/admin" >/dev/null 2>&1
}

restart_admin() {
  if ! systemctl is-enabled --quiet "$ADMIN_SERVICE" 2>/dev/null; then
    return 1
  fi
  sudo -n systemctl restart "$ADMIN_SERVICE" 2>/dev/null || return 1
  # The full status API calls engine HTTP, journalctl and git. Its normal
  # latency can exceed two seconds even when the admin control plane is healthy.
  # Probe the independent process and the actual rendered control page instead.
  for _ in $(seq 1 20); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${ADMIN_PORT}/healthz" >/dev/null 2>&1 &&
       curl -fsS --max-time 3 "http://127.0.0.1:${ADMIN_PORT}/admin" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

recover_admin_control() {
  # The bot may be trading or broker authentication may have expired. Admin
  # recovery does not restart the engine or require broker-flat proof.
  if admin_healthy; then
    return 0
  fi
  log "WARNING: admin page unavailable; restarting only $ADMIN_SERVICE"
  if restart_admin && admin_healthy; then
    log "Admin control plane recovered; trading engine unchanged"
    return 0
  fi
  log "WARNING: admin recovery failed; trading engine unchanged"
  return 1
}

restart_streamlit() {
  if ! systemctl is-enabled --quiet "$STREAMLIT_SERVICE" 2>/dev/null; then
    return 0
  fi
  if sudo -n systemctl restart "$STREAMLIT_SERVICE" 2>/dev/null; then
    :
  else
    pkill -TERM -u "$(id -u)" -f 'streamlit run .*/dashboard/superlite_console.py' 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do
    curl -fsS --max-time 2 "http://127.0.0.1:${STREAMLIT_PORT}/" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

# Manifest polling is delegated to the persistent admin service. The admin only
# requests a transient systemd research service; long research therefore owns an
# independent cgroup and cannot hold the deploy lock or die with an admin restart.
poll_research_request() {
  local code
  code="$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' \
    -X POST \
    -H "Origin: http://127.0.0.1:${ADMIN_PORT}" \
    "http://127.0.0.1:${ADMIN_PORT}/admin/research/poll-manifest" 2>/dev/null || true)"
  case "$code" in
    200|202|409) return 0 ;;
    *) log "WARNING: research manifest poll unavailable (http=${code:-none}); trading deployment unaffected" ;;
  esac
}


# No market-clock blackout: validate and deploy as soon as the account is
# broker-flat, all orders are terminal, and the engine acknowledges an entry
# hold. The release lock and entry-submit lock close the entry race; exits are
# never blocked. No forced deployment on missing broker/engine authority.
intraday_flat_broker_confirmed() {
  local api_key access_token
  api_key="$(first_nonempty_env BROKER_API_KEY ZERODHA_API_KEY KITE_API_KEY 2>/dev/null || true)"
  access_token="$(first_nonempty_env BROKER_ACCESS_TOKEN ZERODHA_ACCESS_TOKEN KITE_ACCESS_TOKEN 2>/dev/null || true)"
  if [ -z "$api_key" ] || [ -z "$access_token" ] || [ ! -x "$VENV/bin/python" ]; then
    log "intraday_release_guard_blocked: broker credentials or Python unavailable"
    return 1
  fi
  BOT_RELEASE_KITE_API_KEY="$api_key" BOT_RELEASE_KITE_ACCESS_TOKEN="$access_token" \
    BOT_RELEASE_LOCAL_PORT="$PORT" BOT_RELEASE_RESTART_PENDING="$RESTART_PENDING_FILE" \
    BOT_RELEASE_VERIFIED_TOKEN_ROTATION="$VERIFIED_TOKEN_ROTATION_FILE" \
    BOT_RELEASE_ENTRY_HOLD_REQUIRED="$RELEASE_HOLD_ACTIVE" \
    "$VENV/bin/python" - <<'PY_FLAT_GUARD'
import hashlib
import json
import os
import sys
import time
import urllib.request

from kiteconnect import KiteConnect

TERMINAL = {"COMPLETE", "CANCELLED", "REJECTED", "LAPSED"}
port = int(os.environ["BOT_RELEASE_LOCAL_PORT"])
kite = KiteConnect(api_key=os.environ["BOT_RELEASE_KITE_API_KEY"], timeout=5)
kite.set_access_token(os.environ["BOT_RELEASE_KITE_ACCESS_TOKEN"])
# A newly verified token exists on disk before the old process can load it.
# This recovery exception is available ONLY for a previously queued restart;
# broker positions and orders still have to be proved flat twice.
restart_marker = os.environ.get("BOT_RELEASE_RESTART_PENDING") or ""
token_rotation_requested = bool(restart_marker and os.path.isfile(restart_marker))
hold_required = os.environ.get("BOT_RELEASE_ENTRY_HOLD_REQUIRED") == "true"

# Only an access token verified with Kite's profile endpoint may bypass an
# *old process* authentication-state check. A generic restart marker is not
# sufficient. External Kite position/order snapshots remain mandatory.
rotation_proof = os.environ.get("BOT_RELEASE_VERIFIED_TOKEN_ROTATION") or ""
token_rotation_verified = False
if token_rotation_requested and rotation_proof and os.path.isfile(rotation_proof):
    try:
        with open(rotation_proof, encoding="utf-8") as proof_stream:
            proof = json.load(proof_stream)
        age = time.time() - float(proof["verified_at"])
        expected = hashlib.sha256(
            os.environ["BOT_RELEASE_KITE_ACCESS_TOKEN"].encode("utf-8")
        ).hexdigest()
        token_rotation_verified = (
            0 <= age <= 12 * 3600
            and str(proof.get("token_sha256") or "") == expected
        )
    except (OSError, ValueError, TypeError, KeyError):
        token_rotation_verified = False

def is_safe() -> bool:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=3) as response:
        live = json.load(response)
    if live.get("engine_http_responsive") is not True:
        return False
    if live.get("quiet") is True:
        # The canonical off-hours control plane has no trading stack and no
        # live entry worker. Broker positions/orders remain authoritative.
        if live.get("bot_loaded") is not False:
            return False
    else:
        if live.get("bot_loaded") is not True:
            return False
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/trading", timeout=3) as response:
            health = json.load(response)
        reconciliation = health.get("reconciliation") or {}
        broker = health.get("broker") or {}
        if hold_required and health.get("release_entry_hold_active") is not True:
            return False
        # A new verified token can make the *old process's* auth state stale.
        # Only this fresh, token-bound proof permits skipping that single
        # local auth bit. Flat broker positions and terminal orders are still
        # verified twice using the new token, with an active entry hold.
        token_rotation_recovery = token_rotation_verified and hold_required
        if (
            health.get("live_orders_armed") is not False
            or reconciliation.get("completed") is not True
            or reconciliation.get("failed") is True
            or reconciliation.get("unprotected_positions")
            or (broker.get("authenticated") is not True and not token_rotation_recovery)
        ):
            return False
    positions = kite.positions()
    orders = kite.orders()
    if (
        not isinstance(positions, dict)
        or not isinstance(positions.get("net"), list)
        or not isinstance(positions.get("day"), list)
        or not isinstance(orders, list)
    ):
        return False
    for row in positions["net"] + positions["day"]:
        if not isinstance(row, dict) or "quantity" not in row:
            return False
        if float(row["quantity"]) != 0:
            return False
    for row in orders:
        if not isinstance(row, dict) or str(row.get("status") or "").upper() not in TERMINAL:
            return False
    return True

try:
    # Both samples must prove no orders/positions and no armed entry path.
    if not is_safe():
        sys.exit(1)
    time.sleep(1)
    if not is_safe():
        sys.exit(1)
except Exception as exc:
    print(f"intraday_release_guard_blocked: {type(exc).__name__}", file=sys.stderr)
    sys.exit(1)
PY_FLAT_GUARD
}


acquire_release_entry_hold() {
  mkdir -p "$CONFIG_DIR"
  if [ -L "$RELEASE_HOLD_FILE" ] || [ -L "$RELEASE_ENTRY_LOCK_FILE" ]; then
    log "release entry hold/lock path is a symlink"
    return 1
  fi
  exec 8>"$RELEASE_ENTRY_LOCK_FILE"
  chmod 600 "$RELEASE_ENTRY_LOCK_FILE"
  # An in-flight LIVE BUY holds a shared lock until the broker call completes.
  # Taking the exclusive lock therefore drains active order submissions.
  if ! flock -w 10 8; then
    log "release entry submission still in flight; retry automatically"
    return 1
  fi
  printf '%s\n' "$BASHPID" > "$RELEASE_HOLD_FILE"
  # Trap cleanup must own the marker even if chmod fails under set -e.
  RELEASE_HOLD_ACTIVE=true
  chmod 600 "$RELEASE_HOLD_FILE"
}

auto_release_should_defer() {
  # An unchanged, healthy, non-restart build needs no broker API calls.
  if [ "$AUTO_MODE" = true ] \
    && [ -n "${BEFORE:-}" ] && [ "$BEFORE" = "${AFTER:-}" ] \
    && [ "${FORCE_RESTART:-false}" = false ] \
    && [ ! -f "${RESTART_PENDING_FILE:-}" ] \
    && service_healthy_confirmed "$BEFORE"; then
    return 0
  fi
  # Pre-CI and pre-validation perform no checkout change or restart. The
  # mandatory broker-flat proof occurs only after the entry hold is acquired.
  if [ "$RELEASE_HOLD_ACTIVE" != true ]; then
    return 1
  fi
  intraday_flat_broker_confirmed && return 1
  return 0
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then log "Another deployment is already running"; exit 0; fi

cd "$APP_DIR"
if [ ! -d .git ]; then
  write_status repository_missing "Git repository missing at $APP_DIR"; exit 1
fi

# Independently heal token-entry controls on every scheduled release check,
# including when broker exposure or a pending GitHub CI run forbids deployment.
# Never change engine state merely because the admin page is unavailable.
if [ "$AUTO_MODE" = true ]; then
  recover_admin_control || true
fi

# A newly fetched release script can run while the checkout still contains a
# legacy setup script. Never pass a repair flag to an installer which cannot
# recognize it: the legacy script would run the ENTIRE bootstrap and restart
# the live bot. A validated candidate can repair permissions before mutation.
control_sudoers_repair_supported() {
  local installer="$1"
  [ -f "$installer" ] && grep -Fqx 'if [ "${1:-}" = "--repair-control-sudoers" ]; then' "$installer"
}

repair_control_sudoers_from() {
  local installer="$1"
  control_sudoers_repair_supported "$installer" || return 1
  bash "$installer" --repair-control-sudoers || return 1
  mkdir -p "$CONFIG_DIR"
  : > "$CONFIG_DIR/control_sudoers_v2_installed"
  log "dashboard/release sudoers permissions repaired"
}

# Repair on a current checkout without launching apt, reinstallation or engine
# restarts. If the checkout predates the repair-only mode, defer this step
# until the candidate has passed release validation.
if [ ! -f "$CONFIG_DIR/control_sudoers_v2_installed" ]; then
  if control_sudoers_repair_supported "$APP_DIR/deploy/lightsail_setup.sh"; then
    repair_control_sudoers_from "$APP_DIR/deploy/lightsail_setup.sh" || \
      log "WARNING: sudoers repair failed; release candidate may need operator recovery"
  else
    log "Legacy setup has no repair-only mode; deferring sudoers repair until validated candidate"
  fi
fi

# Auto mode checks remote revisions without mutating the local runtime.
# Preserve health recovery and independent research polling when a release waits.
if [ "$AUTO_MODE" = true ]; then
  BEFORE="$(git rev-parse HEAD)"
  write_status fetching "checking origin/main from ${BEFORE:0:7}"
  git fetch --quiet origin main || {
    write_status fetch_failed "git fetch failed"; exit 1;
  }
  AFTER="$(git rev-parse origin/main)"
  if auto_release_should_defer; then
    # Never start a different/partially installed revision to recover a live process.
    # A missing or differing runtime provenance requires operator reconciliation.
    current_runtime_sha="$(first_nonempty_env GIT_COMMIT_SHA 2>/dev/null || true)"
    if [[ ! "$current_runtime_sha" =~ ^[0-9a-fA-F]{40}$ ]] || [ "$current_runtime_sha" != "$BEFORE" ]; then
      poll_research_request
      write_status runtime_checkout_mismatch "recovery blocked: checkout ${BEFORE:0:7} differs from confirmed runtime build; manual reconciliation required"
      log "Recovery blocked: do not restart an unverified checkout"
      exit 1
    fi
    live_release_json="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/releasez" 2>/dev/null || true)"
    if [ -n "$live_release_json" ] && ! grep -Eq "\"effective_sha\"[[:space:]]*:[[:space:]]*\"${BEFORE:0:12}\"" <<<"$live_release_json"; then
      poll_research_request
      write_status runtime_checkout_mismatch "recovery blocked: running process does not match checkout ${BEFORE:0:7}"
      log "Recovery blocked: running process and checkout provenance disagree"
      exit 1
    fi
    if ! service_healthy_confirmed "$BEFORE"; then
      # Unknown broker exposure also forbids an automatic recovery restart:
      # the current process may still own a protective exit.
      write_status deferred_unsafe_recovery "engine unhealthy; restart deferred pending broker-flat proof"
      log "Automatic recovery deferred: broker exposure/order state not proven safe"
      exit 0
    fi
    poll_research_request
    if [ "$BEFORE" != "$AFTER" ]; then
      write_status deferred_broker_exposure "pending main ${AFTER:0:7}; deployed ${BEFORE:0:7}; retry when broker-flat with no active orders"
      log "Automatic release deferred: broker exposure, armed engine, or unavailable safety proof"
    else
      if admin_healthy; then
        write_status current "running ${BEFORE:0:7}; no pending revision"
      else
        write_status current_admin_degraded "running ${BEFORE:0:7}; admin service unavailable after recovery attempt"
      fi
    fi
    exit 0
  fi
fi

chmod +x "$APP_DIR/deploy/lightsail_release.sh" 2>/dev/null || true
mkdir -p "$CONFIG_DIR" "$(dirname "$STATUS_FILE")"
chmod 700 "$CONFIG_DIR" 2>/dev/null || true
if [ ! -f "$ENV_FILE" ] && [ -f "$APP_DIR/.env" ]; then
  cp -p "$APP_DIR/.env" "$ENV_FILE"; chmod 600 "$ENV_FILE"
fi
validate_environment
ensure_env_default BOT_RESEARCH_LAUNCHER systemd
# Off-hot-path Supabase defaults are GitHub-owned in Python. Host file
# contains only secrets/operational overrides, not replicated sink settings.
if [ "$RUNTIME_ENV_CHANGED" = true ]; then FORCE_RESTART=true; fi
# Preserve explicit restart requests across failed validation/deployment.
# Exposure no longer defers the default automatic release.
if [ -f "$RESTART_PENDING_FILE" ]; then FORCE_RESTART=true; fi
migrate_systemd_entrypoint
migrate_autodeploy_entrypoint
configure_autodeploy_timer
if [ "$SYSTEMD_ENTRYPOINT_MIGRATED" = true ]; then
  FORCE_RESTART=true
fi

BEFORE="$(git rev-parse HEAD)"
write_status fetching "checking origin/main from ${BEFORE:0:7}"
git fetch --quiet origin main || { write_status fetch_failed "git fetch failed"; exit 1; }
AFTER="$(git rev-parse origin/main)"
current_runtime_sha="$(first_nonempty_env GIT_COMMIT_SHA 2>/dev/null || true)"

# Do not deploy a just-merged main commit until the *push* CI run for that
# exact SHA has completed successfully. PR-head checks do not authorize a
# production release. Unavailable/pending/failed checks retain the running
# revision and are automatically retried on the next timer tick.
if [ "$BEFORE" != "$AFTER" ]; then
  ci_verdict="$("$VENV/bin/python" "$APP_DIR/deploy/scripts/verify_release_ci.py" "$AFTER" 2>/dev/null)" || true
  case "$ci_verdict" in
    passed) log "GitHub main push CI passed for ${AFTER:0:7}" ;;
    pending)
      write_status deferred_ci "main ${AFTER:0:7} awaiting post-merge CI; deployed ${BEFORE:0:7}"
      poll_research_request
      exit 0
      ;;
    failed)
      write_status ci_failed "main ${AFTER:0:7} CI failed; retaining deployed ${BEFORE:0:7}"
      poll_research_request
      exit 1
      ;;
    *)
      write_status ci_unavailable "could not verify main ${AFTER:0:7} CI; retaining deployed ${BEFORE:0:7}"
      poll_research_request
      exit 1
      ;;
  esac
fi

if [ "$BEFORE" = "$AFTER" ] && [ "$FORCE_RESTART" = false ]; then
  if service_healthy_confirmed "$BEFORE"; then
    if [ "$current_runtime_sha" = "$BEFORE" ]; then
      if admin_healthy; then
        write_status current "running ${BEFORE:0:7}"
      else
        write_status current_admin_degraded "running ${BEFORE:0:7}; admin service unavailable after recovery attempt"
      fi
      poll_research_request
      exit 0
    fi
    log "runtime build SHA missing/mismatched; forcing one provenance restart"
    FORCE_RESTART=true
  else
    FORCE_RESTART=true
  fi
fi

CANDIDATE="/tmp/niftybot-candidate-${AFTER:0:12}"
cleanup() {
  # A successfully restarted engine must be able to re-arm normally; a
  # deferred or failed release must also release the temporary entry hold.
  if [ "$RELEASE_HOLD_ACTIVE" = true ]; then
    rm -f "$RELEASE_HOLD_FILE"
    flock -u 8 || true
    RELEASE_HOLD_ACTIVE=false
  fi
  git worktree remove --force "$CANDIDATE" >/dev/null 2>&1 || true
}
trap cleanup EXIT
git worktree prune
rm -rf "$CANDIDATE"
git worktree add --detach --quiet "$CANDIDATE" "$AFTER"
write_status validating "validating ${AFTER:0:7}"

PYTHONPATH="$CANDIDATE/src" "$VENV/bin/python" -m compileall -q "$CANDIDATE/src" || {
  write_status validation_failed "source compile failed for ${AFTER:0:7}"; exit 1;
}
# Validate the canonical console entrypoint but compile *all present* dashboard
# modules. Hard-coding a helper filename here deadlocks deployment after legitimate
# file consolidation: the old validator rejects the new revision containing
# its own fix before it can install that fix.
if [ ! -f "$CANDIDATE/dashboard/superlite_console.py" ]; then
  write_status validation_failed "required dashboard entrypoint missing for ${AFTER:0:7}"
  exit 1
fi
"$VENV/bin/python" -m compileall -q "$CANDIDATE/dashboard" || {
  write_status validation_failed "dashboard compile failed for ${AFTER:0:7}"; exit 1;
}

TARGETED_TESTS=(
  tests/architecture/test_file_header_standard.py
  tests/architecture/test_canonical_bo_ownership.py
  tests/architecture/test_lightsail_release_contract.py
  tests/architecture/test_release_ci_gate.py
  tests/architecture/test_token_rotation_release_guard.py
  tests/core/test_release_entry_handoff.py
  tests/ops/test_research_jobs.py
  tests/scripts/test_archive_kite_history.py
  tests/backtests/test_strategy_bar_research.py
  tests/infra/test_supabase_trade_replication.py
  tests/infra/test_daily_log_archive.py
  tests/infra/test_scheduled_tasks.py
  tests/utils/test_market_hours.py
  tests/core/test_eod_flatten_schedule.py
  tests/strategies/test_candidate_risk_affordability.py
  tests/test_execution_path_contract.py
  tests/execution/test_runtime_order_facade.py
  tests/execution/test_runtime_bracket_facade.py
  tests/execution/test_bracket_persistence_policy.py
  tests/integration/test_canonical_bo_end_to_end.py
  tests/dashboard/test_superlite_event_details.py
  tests/dashboard/test_log_export.py
  tests/data/test_datahub_bounded_persistence.py
  tests/data/test_mdm_tick_coalescing.py
  tests/test_mdm_event_loop_consumer.py
  tests/core/test_selected_option_exec_min_regression.py
  tests/dashboard/test_superlite_admin_core.py
  tests/execution/test_external_close_reconcile.py
)
EXISTING_TESTS=()
for test_path in "${TARGETED_TESTS[@]}"; do
  [ -f "$CANDIDATE/$test_path" ] && EXISTING_TESTS+=("$CANDIDATE/$test_path")
done
if [ "${#EXISTING_TESTS[@]}" -gt 0 ]; then
  env \
    EXECUTION_MODE=SHADOW \
    ENABLE_LIVE=false \
    ENABLE_LIVE_TRADING=false \
    ORDERS__ENABLE_LIVE=false \
    PAPER_MODE=true \
    SHADOW_MODE=true \
    DATA_DIR="$CANDIDATE/.validation-data" \
    PYTHONPATH="$CANDIDATE/src:$CANDIDATE" \
    "$VENV/bin/python" -m pytest -q "${EXISTING_TESTS[@]}" || {
    write_status validation_failed "focused tests failed for ${AFTER:0:7}"; exit 1;
  }
fi

# Acquire the exclusive entry-submit barrier only after validation. Block
# new BUY submissions while allowing protective SELL exits; then confirm two
# independently fresh broker position/order snapshots at any market hour.
if ! acquire_release_entry_hold; then
  write_status deferred_entry_handoff "candidate ${AFTER:0:7} validated; entry submission busy; retry automatically"
  exit 0
fi
if auto_release_should_defer; then
  if [ "$FORCE_RESTART" = true ]; then
    mkdir -p "$CONFIG_DIR"
    : > "$RESTART_PENDING_FILE"
  fi
  poll_research_request
  write_status deferred_broker_exposure "candidate ${AFTER:0:7} validated; checkout untouched until broker-flat, terminal orders and engine hold acknowledgement"
  exit 0
fi

# Permission repair is required before checkout changes or any service
# restart. Use only the *validated candidate* installer; never run the legacy
# full-setup script with a flag it does not understand.
if [ ! -f "$CONFIG_DIR/control_sudoers_v2_installed" ]; then
  if ! repair_control_sudoers_from "$CANDIDATE/deploy/lightsail_setup.sh"; then
    write_status control_permissions_failed "validated release blocked: restricted sudoers repair failed; operator recovery required"
    exit 1
  fi
fi

mkdir -p "$CONFIG_DIR/backups"
cp -p "$ENV_FILE" "$CONFIG_DIR/backups/niftybot.env.$(date +%Y%m%d_%H%M%S)"
write_status deploying "deploying ${AFTER:0:7}"
[ "$BEFORE" = "$AFTER" ] || git reset --hard --quiet "$AFTER"

# Most updates only change GitHub-owned Python code. The established
# editable install points at APP_DIR/src and needs no pip operation unless
# packaging/dependencies change or the installed package is missing. Avoid
# mutating the running interpreter's site-packages for ordinary commits.
if BOT_DEPLOY_APP_DIR="$APP_DIR" "$VENV/bin/python" -c 'from importlib.util import find_spec; from pathlib import Path; import os; spec = find_spec("nifty_scalper_bot"); assert spec is not None and spec.origin and Path(spec.origin).resolve().is_relative_to(Path(os.environ["BOT_DEPLOY_APP_DIR"]).resolve() / "src")' >/dev/null 2>&1 \
  && git diff --quiet "$BEFORE" "$AFTER" -- \
    pyproject.toml setup.cfg setup.py requirements.txt 'requirements*.txt' \
    requirements/ uv.lock poetry.lock Pipfile Pipfile.lock; then
  log "Skipping pip install: installed editable package and unchanged packaging"
else
  PACKAGE_INSTALL_ATTEMPTED=true
  if ! "$VENV/bin/python" -m pip install --quiet -e .; then
    if [ "$BEFORE" != "$AFTER" ]; then
      git reset --hard --quiet "$BEFORE"; "$VENV/bin/python" -m pip install --quiet -e . || true
    fi
    write_status install_failed "install failed; restored ${BEFORE:0:7}"; exit 1
  fi
fi

if [ -x "$STREAMLIT_VENV/bin/python" ] && [ -f dashboard/requirements.txt ]; then
  # Same-revision recovery and code-only changes must not mutate console
  # packages. Only an actual requirements diff authorizes a refresh.
  if [ "$BEFORE" != "$AFTER" ] && ! git diff --quiet "$BEFORE" "$AFTER" -- dashboard/requirements.txt; then
    if ! "$STREAMLIT_VENV/bin/python" -m pip install --quiet -r dashboard/requirements.txt; then
      STREAMLIT_DEPENDENCY_FAILED=true
      log "ERROR: dashboard dependency refresh failed; trading engine release continues, console health is unverified"
    fi
  fi
fi

if ! set_runtime_build_sha "$AFTER"; then
  if [ "$BEFORE" != "$AFTER" ]; then
    git reset --hard --quiet "$BEFORE"
    if [ "$PACKAGE_INSTALL_ATTEMPTED" = true ]; then
      "$VENV/bin/python" -m pip install --quiet -e . || true
    fi
  fi
  write_status env_update_failed "could not publish runtime build SHA ${AFTER:0:7}"
  exit 1
fi
# Final broker/arming boundary check before automatic restart, regardless of
# market hours. Restore the prior checkout if the proof no longer holds.
if auto_release_should_defer; then
  if [ "$FORCE_RESTART" = true ]; then
    mkdir -p "$CONFIG_DIR"
    : > "$RESTART_PENDING_FILE"
  fi
  if [ "$BEFORE" != "$AFTER" ]; then
    if ! git reset --hard --quiet "$BEFORE"; then
      write_status runtime_restore_failed "checkout restoration failed; candidate may be partially installed"
      exit 1
    fi
    if [ "$PACKAGE_INSTALL_ATTEMPTED" = true ]; then
      if ! "$VENV/bin/python" -m pip install --quiet -e .; then
        write_status runtime_restore_failed "retained revision dependency restoration failed; manual recovery required"
        exit 1
      fi
    fi
  fi
  if ! set_runtime_build_sha "$BEFORE"; then
    write_status runtime_restore_failed "retained revision build provenance restoration failed; manual recovery required"
    exit 1
  fi
  write_status deferred_broker_exposure "broker/holding state changed before engine restart; current revision ${BEFORE:0:7} restored"
  poll_research_request
  exit 0
fi
sudo systemctl restart "$SERVICE"
if wait_for_service "$AFTER"; then
  rm -f "$RESTART_PENDING_FILE" "$VERIFIED_TOKEN_ROTATION_FILE"
  admin_ready=false
  if restart_admin; then
    admin_ready=true
  else
    log "WARNING: admin service restart/health check failed; research manifest launch skipped"
  fi
  if broker_auth_dependency_degraded; then
    if [ "$STREAMLIT_DEPENDENCY_FAILED" = true ]; then
      log "ERROR: Streamlit restart skipped after failed dependency refresh; broker auth also degraded"
    else
      restart_streamlit || log "WARNING: dashboard restart failed while broker auth is degraded"
    fi
    if [ "$admin_ready" = true ]; then
      if [ "$STREAMLIT_DEPENDENCY_FAILED" = true ]; then
        write_status deployed_broker_and_console_dependencies_degraded "deployed ${AFTER:0:7}; broker authentication unavailable; console dependency refresh failed"
      else
        write_status deployed_dependency_degraded "deployed ${AFTER:0:7}; engine healthy, broker authentication unavailable; trading blocked"
      fi
    else
      if [ "$STREAMLIT_DEPENDENCY_FAILED" = true ]; then
        write_status deployed_admin_broker_and_console_dependencies_degraded "deployed ${AFTER:0:7}; admin unavailable; broker auth and console dependencies degraded"
      else
        write_status deployed_admin_and_broker_degraded "deployed ${AFTER:0:7}; engine healthy, admin service unavailable, broker authentication unavailable"
      fi
    fi
    logger -t niftybot-deploy "candidate deployed; broker authentication unavailable, trading remains blocked"
  elif [ "$STREAMLIT_DEPENDENCY_FAILED" = true ]; then
    # Console is ancillary; report degraded instead of blocking healthy
    # protective trading paths or claiming unverified console availability.
    if [ "$admin_ready" = true ]; then
      write_status deployed_console_dependencies_degraded "deployed ${AFTER:0:7}; bot and admin healthy; console dependency refresh failed"
    else
      write_status deployed_admin_console_dependencies_degraded "deployed ${AFTER:0:7}; bot healthy; admin unavailable and console dependency refresh failed"
    fi
    logger -t niftybot-deploy "console dependency installation failed; engine remains available"
  elif restart_streamlit; then
    if [ "$admin_ready" = true ]; then
      write_status deployed "deployed ${AFTER:0:7}; bot, admin and console healthy"
    else
      write_status deployed_admin_degraded "deployed ${AFTER:0:7}; bot and console healthy, admin service unavailable"
    fi
  else
    if [ "$admin_ready" = true ]; then
      write_status deployed_console_degraded "deployed ${AFTER:0:7}; bot and admin healthy, console restart failed"
    else
      write_status deployed_controls_degraded "deployed ${AFTER:0:7}; engine healthy, admin and console unavailable"
    fi
    logger -t niftybot-deploy "bot deployed; Streamlit health check failed"
  fi
  logger -t niftybot-deploy "validated and deployed ${BEFORE:0:7} -> ${AFTER:0:7}"
  if [ "$admin_ready" = true ]; then
    poll_research_request
  fi
  exit 0
fi

if [ "$BEFORE" != "$AFTER" ]; then
  git reset --hard --quiet "$BEFORE"
  if [ "$PACKAGE_INSTALL_ATTEMPTED" = true ]; then
    "$VENV/bin/python" -m pip install --quiet -e . || true
  fi
  set_runtime_build_sha "$BEFORE" || true
  sudo systemctl restart "$SERVICE"
  restart_admin || true
  restart_streamlit || true
  write_status rolled_back "bot health check failed; restored ${BEFORE:0:7}"
else
  write_status restart_failed "current revision failed readiness after restart"
fi
if [ "$AUTO_MODE" = false ]; then sudo systemctl status "$SERVICE" --no-pager -l || true; fi
exit 1
