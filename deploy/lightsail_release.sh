#!/usr/bin/env bash
# File purpose: Validate and deploy the production AWS Lightsail release.
# Key responsibilities: Preserve host secrets, validate candidate code, restart systemd services, verify readiness, and roll back failed bot revisions.
# Operational constraints: Secrets live outside the Git checkout; only one deployment runs at a time; Streamlit failure must not interrupt a healthy trading engine.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
SERVICE="${BOT_SERVICE_NAME:-niftybot}"
STREAMLIT_SERVICE="${BOT_STREAMLIT_SERVICE_NAME:-niftybot-streamlit}"
PORT="${PORT:-8080}"
STREAMLIT_PORT="${BOT_STREAMLIT_PORT:-8501}"
CONFIG_DIR="${NIFTYBOT_CONFIG_DIR:-/home/ubuntu/.config/niftybot}"
ENV_FILE="${BOT_ENV_FILE:-$CONFIG_DIR/niftybot.env}"
STATUS_FILE="${BOT_UPDATE_STATUS_FILE:-$APP_DIR/data/auto_update_status.json}"
VENV="${BOT_VENV:-$APP_DIR/.venv}"
STREAMLIT_VENV="${BOT_STREAMLIT_VENV:-$APP_DIR/.streamlit-venv}"
LOCK_FILE="${BOT_DEPLOY_LOCK_FILE:-/tmp/niftybot-deploy.lock}"
FORCE_RESTART=false
AUTO_MODE=false
SYSTEMD_ENTRYPOINT_MIGRATED=false

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

service_healthy() {
  local live_json
  live_json="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/livez" 2>/dev/null || true)"
  grep -Eq '"bot_loaded"[[:space:]]*:[[:space:]]*true' <<<"$live_json" || return 1
  if env_truthy ENABLE_LIVE; then
    curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1 || return 1
  fi
}

wait_for_service() {
  for _ in $(seq 1 45); do service_healthy && return 0; sleep 2; done
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
  log "migrated $SERVICE ExecStart to deployment_main:app; EnvironmentFile preserved"
}

restart_streamlit() {
  if ! systemctl is-enabled --quiet "$STREAMLIT_SERVICE" 2>/dev/null; then
    return 0
  fi
  if sudo -n systemctl restart "$STREAMLIT_SERVICE" 2>/dev/null; then
    :
  else
    # Existing hosts may not yet have the expanded sudoers rule. The process is
    # owned by ubuntu and Restart=always, so TERM safely asks systemd to relaunch it.
    pkill -TERM -u "$(id -u)" -f 'streamlit run .*/dashboard/superlite_console.py' 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do
    curl -fsS --max-time 2 "http://127.0.0.1:${STREAMLIT_PORT}/" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then log "Another deployment is already running"; exit 0; fi

cd "$APP_DIR"
if [ ! -d .git ]; then
  write_status repository_missing "Git repository missing at $APP_DIR"; exit 1
fi
mkdir -p "$CONFIG_DIR" "$(dirname "$STATUS_FILE")"
chmod 700 "$CONFIG_DIR" 2>/dev/null || true
if [ ! -f "$ENV_FILE" ] && [ -f "$APP_DIR/.env" ]; then
  cp -p "$APP_DIR/.env" "$ENV_FILE"; chmod 600 "$ENV_FILE"
fi
validate_environment
migrate_systemd_entrypoint
if [ "$SYSTEMD_ENTRYPOINT_MIGRATED" = true ]; then
  FORCE_RESTART=true
fi

BEFORE="$(git rev-parse HEAD)"
write_status fetching "checking origin/main from ${BEFORE:0:7}"
git fetch --quiet origin main || { write_status fetch_failed "git fetch failed"; exit 1; }
AFTER="$(git rev-parse origin/main)"

if [ "$BEFORE" = "$AFTER" ] && [ "$FORCE_RESTART" = false ]; then
  if service_healthy; then write_status current "running ${BEFORE:0:7}"; exit 0; fi
  FORCE_RESTART=true
fi

CANDIDATE="/tmp/niftybot-candidate-${AFTER:0:12}"
cleanup() { git worktree remove --force "$CANDIDATE" >/dev/null 2>&1 || true; }
trap cleanup EXIT
git worktree prune
rm -rf "$CANDIDATE"
git worktree add --detach --quiet "$CANDIDATE" "$AFTER"
write_status validating "validating ${AFTER:0:7}"

PYTHONPATH="$CANDIDATE/src" "$VENV/bin/python" -m compileall -q "$CANDIDATE/src" || {
  write_status validation_failed "source compile failed for ${AFTER:0:7}"; exit 1;
}
"$VENV/bin/python" -m py_compile \
  "$CANDIDATE/dashboard/event_buffer.py" \
  "$CANDIDATE/dashboard/log_export.py" \
  "$CANDIDATE/dashboard/superlite_console.py" || {
  write_status validation_failed "dashboard compile failed for ${AFTER:0:7}"; exit 1;
}

TARGETED_TESTS=(
  tests/architecture/test_file_header_standard.py
  tests/architecture/test_canonical_bo_ownership.py
  tests/architecture/test_lightsail_release_contract.py
  tests/test_execution_path_contract.py
  tests/execution/test_runtime_order_facade.py
  tests/execution/test_runtime_bracket_facade.py
  tests/integration/test_canonical_bo_end_to_end.py
  tests/dashboard/test_event_buffer_truth.py
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
  PYTHONPATH="$CANDIDATE/src:$CANDIDATE" "$VENV/bin/python" -m pytest -q "${EXISTING_TESTS[@]}" || {
    write_status validation_failed "focused tests failed for ${AFTER:0:7}"; exit 1;
  }
fi

mkdir -p "$CONFIG_DIR/backups"
cp -p "$ENV_FILE" "$CONFIG_DIR/backups/niftybot.env.$(date +%Y%m%d_%H%M%S)"
write_status deploying "deploying ${AFTER:0:7}"
[ "$BEFORE" = "$AFTER" ] || git reset --hard --quiet "$AFTER"

if ! "$VENV/bin/python" -m pip install --quiet -e .; then
  if [ "$BEFORE" != "$AFTER" ]; then
    git reset --hard --quiet "$BEFORE"; "$VENV/bin/python" -m pip install --quiet -e . || true
  fi
  write_status install_failed "install failed; restored ${BEFORE:0:7}"; exit 1
fi

if [ -x "$STREAMLIT_VENV/bin/python" ] && [ -f dashboard/requirements.txt ]; then
  if [ "$BEFORE" = "$AFTER" ] || ! git diff --quiet "$BEFORE" "$AFTER" -- dashboard/requirements.txt; then
    "$STREAMLIT_VENV/bin/python" -m pip install --quiet -r dashboard/requirements.txt || \
      log "WARNING: dashboard dependency refresh failed"
  fi
fi

sudo systemctl restart "$SERVICE"
if wait_for_service; then
  if restart_streamlit; then
    write_status deployed "deployed ${AFTER:0:7}; bot and console healthy"
  else
    write_status deployed_console_degraded "deployed ${AFTER:0:7}; bot healthy, console restart failed"
    logger -t niftybot-deploy "bot deployed; Streamlit health check failed"
  fi
  logger -t niftybot-deploy "validated and deployed ${BEFORE:0:7} -> ${AFTER:0:7}"
  exit 0
fi

if [ "$BEFORE" != "$AFTER" ]; then
  git reset --hard --quiet "$BEFORE"
  "$VENV/bin/python" -m pip install --quiet -e . || true
  sudo systemctl restart "$SERVICE"
  restart_streamlit || true
  write_status rolled_back "bot health check failed; restored ${BEFORE:0:7}"
else
  write_status restart_failed "current revision failed readiness after restart"
fi
if [ "$AUTO_MODE" = false ]; then sudo systemctl status "$SERVICE" --no-pager -l || true; fi
exit 1
