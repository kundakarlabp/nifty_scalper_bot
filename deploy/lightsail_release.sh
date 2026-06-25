#!/usr/bin/env bash
# File purpose: Validate and deploy the production AWS Lightsail release.
# Key responsibilities: Preserve host secrets, validate candidate code, restart systemd, verify readiness, and roll back failed code revisions.
# Operational constraints: Secrets live outside the Git checkout; only one deployment runs at a time; LIVE releases must pass /readyz.
set -euo pipefail

APP_DIR="${BOT_APP_DIR:-/home/ubuntu/nifty_scalper_bot}"
SERVICE="${BOT_SERVICE_NAME:-niftybot}"
PORT="${PORT:-8080}"
CONFIG_DIR="${NIFTYBOT_CONFIG_DIR:-/home/ubuntu/.config/niftybot}"
ENV_FILE="${BOT_ENV_FILE:-$CONFIG_DIR/niftybot.env}"
STATUS_FILE="${BOT_UPDATE_STATUS_FILE:-$APP_DIR/data/auto_update_status.json}"
VENV="${BOT_VENV:-$APP_DIR/.venv}"
LOCK_FILE="${BOT_DEPLOY_LOCK_FILE:-/tmp/niftybot-deploy.lock}"
FORCE_RESTART=false
AUTO_MODE=false

for arg in "$@"; do
  case "$arg" in
    --force|--force-restart) FORCE_RESTART=true ;;
    --auto) AUTO_MODE=true ;;
    *) printf '[lightsail-release] unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

log() {
  printf '[lightsail-release] %s\n' "$*"
}

write_status() {
  local state="$1"
  local message="$2"
  local escaped
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
    if [ -n "${value//[[:space:]]/}" ]; then
      printf '%s' "$value"
      return 0
    fi
  done
  return 1
}

env_truthy() {
  local key="$1" value
  value="$(first_nonempty_env "$key" 2>/dev/null || true)"
  case "${value,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

validate_environment() {
  if [ ! -f "$ENV_FILE" ]; then
    write_status env_missing "environment file missing: $ENV_FILE"
    log "ERROR: environment file missing: $ENV_FILE"
    return 1
  fi
  chmod 600 "$ENV_FILE" 2>/dev/null || true

  if ! first_nonempty_env BROKER_API_KEY ZERODHA_API_KEY KITE_API_KEY >/dev/null; then
    write_status env_invalid "broker API key is missing"
    log "ERROR: broker API key is missing"
    return 1
  fi
  if ! first_nonempty_env BROKER_API_SECRET ZERODHA_API_SECRET KITE_API_SECRET >/dev/null; then
    write_status env_invalid "broker API secret is missing"
    log "ERROR: broker API secret is missing"
    return 1
  fi

  if env_truthy ENABLE_LIVE; then
    local execution_mode
    execution_mode="$(first_nonempty_env EXECUTION_MODE 2>/dev/null || true)"
    if [ "${execution_mode^^}" != "LIVE" ]; then
      write_status env_invalid "ENABLE_LIVE=true requires EXECUTION_MODE=LIVE"
      log "ERROR: inconsistent live execution flags"
      return 1
    fi
    if ! first_nonempty_env BROKER_ACCESS_TOKEN ZERODHA_ACCESS_TOKEN KITE_ACCESS_TOKEN >/dev/null; then
      write_status env_invalid "LIVE mode requires a broker access token"
      log "ERROR: LIVE mode requires a broker access token"
      return 1
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
  return 0
}

wait_for_service() {
  local attempt
  for attempt in $(seq 1 45); do
    if service_healthy; then
      return 0
    fi
    sleep 2
  done
  return 1
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another deployment is already running"
  exit 0
fi

cd "$APP_DIR"
if [ ! -d .git ]; then
  write_status repository_missing "Git repository missing at $APP_DIR"
  log "ERROR: Git repository missing at $APP_DIR"
  exit 1
fi

mkdir -p "$CONFIG_DIR" "$(dirname "$STATUS_FILE")"
chmod 700 "$CONFIG_DIR" 2>/dev/null || true

# One-time migration from the historic in-repository .env location.
if [ ! -f "$ENV_FILE" ] && [ -f "$APP_DIR/.env" ]; then
  cp -p "$APP_DIR/.env" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  log "Migrated environment file outside the Git checkout"
fi

validate_environment

BEFORE="$(git rev-parse HEAD)"
write_status fetching "checking origin/main from ${BEFORE:0:7}"
if ! git fetch --quiet origin main; then
  write_status fetch_failed "git fetch failed"
  exit 1
fi
AFTER="$(git rev-parse origin/main)"

if [ "$BEFORE" = "$AFTER" ] && [ "$FORCE_RESTART" = false ]; then
  if service_healthy; then
    write_status current "running ${BEFORE:0:7}"
    exit 0
  fi
  log "Current revision is not healthy; forcing a validated restart"
  FORCE_RESTART=true
fi

CANDIDATE="/tmp/niftybot-candidate-${AFTER:0:12}"
cleanup() {
  git worktree remove --force "$CANDIDATE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git worktree prune
rm -rf "$CANDIDATE"
git worktree add --detach --quiet "$CANDIDATE" "$AFTER"
write_status validating "validating ${AFTER:0:7}"

if ! PYTHONPATH="$CANDIDATE/src" "$VENV/bin/python" -m compileall -q "$CANDIDATE/src"; then
  write_status validation_failed "compile failed for ${AFTER:0:7}"
  exit 1
fi

TARGETED_TESTS=(
  tests/architecture/test_file_header_standard.py
  tests/architecture/test_canonical_bo_ownership.py
  tests/architecture/test_lightsail_release_contract.py
  tests/test_execution_path_contract.py
  tests/execution/test_runtime_order_facade.py
  tests/execution/test_runtime_bracket_facade.py
  tests/integration/test_canonical_bo_end_to_end.py
)
EXISTING_TESTS=()
for test_path in "${TARGETED_TESTS[@]}"; do
  [ -f "$CANDIDATE/$test_path" ] && EXISTING_TESTS+=("$CANDIDATE/$test_path")
done
if [ "${#EXISTING_TESTS[@]}" -gt 0 ]; then
  if ! PYTHONPATH="$CANDIDATE/src" "$VENV/bin/python" -m pytest -q "${EXISTING_TESTS[@]}"; then
    write_status validation_failed "focused tests failed for ${AFTER:0:7}"
    exit 1
  fi
fi

mkdir -p "$CONFIG_DIR/backups"
cp -p "$ENV_FILE" "$CONFIG_DIR/backups/niftybot.env.$(date +%Y%m%d_%H%M%S)"

write_status deploying "deploying ${AFTER:0:7}"
if [ "$BEFORE" != "$AFTER" ]; then
  git reset --hard --quiet "$AFTER"
fi

if ! "$VENV/bin/python" -m pip install --quiet -e .; then
  if [ "$BEFORE" != "$AFTER" ]; then
    git reset --hard --quiet "$BEFORE"
    "$VENV/bin/python" -m pip install --quiet -e . || true
  fi
  write_status install_failed "install failed; restored ${BEFORE:0:7}"
  exit 1
fi

sudo systemctl restart "$SERVICE"
if wait_for_service; then
  write_status deployed "deployed ${AFTER:0:7}"
  logger -t niftybot-deploy "validated and deployed ${BEFORE:0:7} -> ${AFTER:0:7}"
  exit 0
fi

if [ "$BEFORE" != "$AFTER" ]; then
  git reset --hard --quiet "$BEFORE"
  "$VENV/bin/python" -m pip install --quiet -e . || true
  sudo systemctl restart "$SERVICE"
  write_status rolled_back "health check failed; restored ${BEFORE:0:7}"
  logger -t niftybot-deploy "health check failed; restored ${AFTER:0:7} -> ${BEFORE:0:7}"
else
  write_status restart_failed "current revision failed readiness after restart"
fi

if [ "$AUTO_MODE" = false ]; then
  sudo systemctl status "$SERVICE" --no-pager -l || true
fi
exit 1
