"""Architecture checks for the AWS Lightsail release and secret boundary."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dotenv_is_not_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
    assert ".env" in _text(".gitignore").splitlines()
    assert (ROOT / ".env.example").is_file()


def test_lightsail_uses_external_environment_file() -> None:
    setup = _text("deploy/lightsail_setup.sh")
    assert 'CONFIG_DIR="/home/ubuntu/.config/niftybot"' in setup
    assert 'ENV_FILE="$CONFIG_DIR/niftybot.env"' in setup
    assert "EnvironmentFile=$ENV_FILE" in setup
    assert 'ln -sfn "$ENV_FILE" "$LEGACY_ENV"' in setup
    assert "DEPLOYMENT_PLATFORM=aws_lightsail" in setup
    assert "nifty_scalper_bot.deployment_main:app" in setup
    assert (
        "ExecStart=/usr/bin/env bash $APP_DIR/deploy/lightsail_release.sh --auto"
        in setup
    )


def test_release_runner_validates_and_rolls_back() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert "flock -n 9" in release
    assert "git worktree add" in release
    assert "compileall" in release
    assert "pytest" in release
    assert "tests/execution/test_bracket_persistence_policy.py" in release
    assert "tests/data/test_datahub_bounded_persistence.py" in release
    assert "tests/data/test_mdm_tick_coalescing.py" in release
    assert "tests/test_mdm_event_loop_consumer.py" in release
    assert "dashboard/superlite_console.py" in release
    assert "dashboard/operations_console.py" not in release
    assert '"bot_loaded"[[:space:]]*:[[:space:]]*true' in release
    assert '"engine_http_responsive"[[:space:]]*:[[:space:]]*true' in release
    assert '"quiet"[[:space:]]*:[[:space:]]*true' in release
    assert '"bot_loaded"[[:space:]]*:[[:space:]]*false' in release
    assert "http://127.0.0.1:${PORT}/livez" in release
    assert "http://127.0.0.1:${PORT}/releasez" in release
    assert "broker_auth_dependency_degraded" in release
    health_block = release.split("service_healthy", 1)[1].split("wait_for_service", 1)[
        0
    ]
    assert "/readyz" not in health_block
    assert 'git reset --hard --quiet "$BEFORE"' in release
    assert 'sudo systemctl restart "$SERVICE"' in release


def test_release_health_accepts_only_loaded_or_explicit_quiet_runtime() -> None:
    release = _text("deploy/lightsail_release.sh")
    health = release.split("service_healthy", 1)[1].split("wait_for_service", 1)[0]
    engine = health.index('"engine_http_responsive"')
    loaded = health.index('"bot_loaded"', engine)
    quiet = health.index('"quiet"', loaded)
    unloaded = health.index('"bot_loaded"', quiet)
    assert engine < loaded < quiet < unloaded
    assert "return 0" in health[loaded:quiet]
    assert "broker_auth_dependency_degraded" in health[quiet:]


def test_lightsail_release_migrates_existing_systemd_entrypoint_safely() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert "migrate_systemd_entrypoint" in release
    assert "nifty_scalper_bot.deployment_main:app" in release
    assert "sudo systemctl daemon-reload" in release
    migration_block = release.split("migrate_systemd_entrypoint", 1)[1].split(
        "migrate_autodeploy_entrypoint", 1
    )[0]
    assert "EnvironmentFile" not in migration_block
    assert "ExecStart=" in migration_block
    assert "SYSTEMD_ENTRYPOINT_MIGRATED=true" in migration_block
    assert "sudo systemctl daemon-reload" in migration_block


def test_lightsail_release_migrates_autodeploy_entrypoint_to_bash() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert "migrate_autodeploy_entrypoint" in release
    assert (
        'AUTODEPLOY_SERVICE="${BOT_AUTODEPLOY_SERVICE_NAME:-niftybot-autodeploy}"'
        in release
    )
    migration_block = release.split("migrate_autodeploy_entrypoint", 1)[1].split(
        "restart_streamlit", 1
    )[0]
    assert (
        "ExecStart=/usr/bin/env bash ${APP_DIR}/deploy/lightsail_release.sh --auto"
        in migration_block
    )
    assert "AUTODEPLOY_ENTRYPOINT_MIGRATED=true" in migration_block
    assert "sudo systemctl daemon-reload" in migration_block


def test_lightsail_migration_forces_restart_before_healthy_no_change_exit() -> None:
    release = _text("deploy/lightsail_release.sh")
    migration_call = release.index("migrate_systemd_entrypoint")
    force_restart = release.index("FORCE_RESTART=true", migration_call)
    healthy_no_change = release.index('if [ "$BEFORE" = "$AFTER" ]', force_restart)
    assert migration_call < force_restart < healthy_no_change
    candidate_index = release.index("CANDIDATE=", healthy_no_change)
    no_change_block = release[healthy_no_change:candidate_index]
    assert '[ "$FORCE_RESTART" = false ]' in no_change_block


def test_unchanged_release_confirms_health_failure_before_restart() -> None:
    release = _text("deploy/lightsail_release.sh")
    helper = release.split("service_healthy_confirmed()", 1)[1].split(
        "wait_for_service()", 1
    )[0]

    assert "BOT_DEPLOY_HEALTH_PROBE_ATTEMPTS:-3" in helper
    assert "BOT_DEPLOY_HEALTH_PROBE_INTERVAL_SEC:-2" in helper
    assert 'service_healthy "$expected_sha"' in helper
    assert "for attempt in" in helper

    unchanged = release.split('if [ "$BEFORE" = "$AFTER" ]', 1)[1].split(
        "CANDIDATE=", 1
    )[0]
    assert 'service_healthy_confirmed "$BEFORE"' in unchanged
    assert 'service_healthy "$BEFORE"' not in unchanged


def test_changed_release_restarts_current_revision_admin_before_research_poll() -> None:
    release = _text("deploy/lightsail_release.sh")

    assert 'ADMIN_SERVICE="${BOT_ADMIN_SERVICE_NAME:-niftybot-admin}"' in release
    assert "restart_admin()" in release
    helper = release.split("restart_admin()", 1)[1].split("restart_streamlit()", 1)[0]
    assert 'systemctl restart "$ADMIN_SERVICE"' in helper
    assert "http://127.0.0.1:${ADMIN_PORT}/admin/api/status" in helper

    deployed = release.split('if wait_for_service "$AFTER"; then', 1)[1].split(
        "exit 0", 1
    )[0]
    admin_restart = deployed.index("restart_admin")
    research_poll = deployed.index("poll_research_request")
    assert admin_restart < research_poll
    assert "admin_ready=true" in deployed
    assert 'if [ "$admin_ready" = true ]; then' in deployed


def test_rollback_restarts_admin_on_restored_revision() -> None:
    release = _text("deploy/lightsail_release.sh")
    rollback = release.split('if [ "$BEFORE" != "$AFTER" ]; then', 2)[-1]
    reset = rollback.index('git reset --hard --quiet "$BEFORE"')
    engine_restart = rollback.index('sudo systemctl restart "$SERVICE"', reset)
    admin_restart = rollback.index("restart_admin", engine_restart)
    assert reset < engine_restart < admin_restart


def test_release_enables_independent_research_systemd_launcher() -> None:
    release = _text("deploy/lightsail_release.sh")

    assert "ensure_env_default BOT_RESEARCH_LAUNCHER systemd" in release
    assert "transient systemd research service" in release
    block = release.split("poll_research_request()", 1)[1].split("exec 9>", 1)[0]
    assert "run_research_job.py" not in block


def test_release_delegates_research_manifest_to_persistent_admin() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert 'ADMIN_PORT="${BOT_ADMIN_PORT:-8081}"' in release
    block = release.split("poll_research_request()", 1)[1].split("exec 9>", 1)[0]

    assert "/admin/research/poll-manifest" in block
    assert "http://127.0.0.1:${ADMIN_PORT}" in block
    assert "--max-time 5" in block
    assert "run_research_job.py" not in block
    assert "200|202|409" in block


def test_deploy_helpers_do_not_embed_credentials() -> None:
    paths = (
        "deploy/lightsail_setup.sh",
        "deploy/lightsail_release.sh",
        "deploy/scripts/update_and_deploy.sh",
        "ops/redeploy.sh",
    )
    assignment = re.compile(
        r"(?m)^(?:KITE|ZERODHA|BROKER)_(?:API_KEY|API_SECRET|ACCESS_TOKEN)=([^\s]*)$"
    )
    for path in paths:
        text = _text(path)
        for match in assignment.finditer(text):
            assert match.group(1) == "", f"credential literal found in {path}"


def test_legacy_redeploy_entrypoints_delegate_to_canonical_runner() -> None:
    for path in ("deploy/scripts/update_and_deploy.sh", "ops/redeploy.sh"):
        text = _text(path)
        assert "deploy/lightsail_release.sh" in text
        assert "git reset --hard origin/main" not in text
        assert "docker build" not in text


def test_release_runner_publishes_exact_runtime_build_sha() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert "set_runtime_build_sha()" in release
    assert 'current_runtime_sha="$(first_nonempty_env GIT_COMMIT_SHA' in release

    deploy = release.index('write_status deploying "deploying ${AFTER:0:7}"')
    publish_after = release.index('set_runtime_build_sha "$AFTER"', deploy)
    restart_after = release.index('sudo systemctl restart "$SERVICE"', publish_after)
    assert deploy < publish_after < restart_after

    rollback = release.index('git reset --hard --quiet "$BEFORE"', restart_after)
    publish_before = release.index('set_runtime_build_sha "$BEFORE"', rollback)
    restart_before = release.index('sudo systemctl restart "$SERVICE"', publish_before)
    assert rollback < publish_before < restart_before

    healthy_no_change = release.index('if [ "$BEFORE" = "$AFTER" ]')
    candidate = release.index("CANDIDATE=", healthy_no_change)
    no_change_block = release[healthy_no_change:candidate]
    assert "current_runtime_sha" in no_change_block
    assert "FORCE_RESTART=true" in no_change_block


def test_release_enables_existing_daily_log_archive() -> None:
    release = _text("deploy/lightsail_release.sh")
    setup = _text("deploy/lightsail_setup.sh")
    example = _text(".env.example")
    endpoint = (
        "https://dehdptgkqbrkyzyodicd.supabase.co/functions/v1/nifty-log-archive-ingest"
    )

    assert "set_env_value SUPABASE_LOG_ARCHIVE_ENABLED true" in release
    for text in (release, setup):
        assert endpoint in text
        assert "SUPABASE_LOG_ARCHIVE_INTERVAL_SECONDS 300" in text
    assert "SUPABASE_LOG_ARCHIVE_ENABLED=true" in example


def test_release_provisions_nonsecret_trade_replication_settings() -> None:
    release = _text("deploy/lightsail_release.sh")
    setup = _text("deploy/lightsail_setup.sh")
    endpoint = (
        "https://dehdptgkqbrkyzyodicd.supabase.co/functions/v1/nifty-trade-ingest"
    )
    for text in (release, setup):
        assert "SUPABASE_TRADE_REPLICATION_ENABLED true" in text
        assert endpoint in text
        assert "SUPABASE_SERVICE_ROLE" not in text
        assert "SUPABASE_SECRET" not in text
    assert (
        'if [ "$RUNTIME_ENV_CHANGED" = true ]; then FORCE_RESTART=true; fi' in release
    )
    assert "tests/infra/test_supabase_trade_replication.py" in release


def test_release_health_separates_candidate_integrity_from_broker_auth() -> None:
    release = _text("deploy/lightsail_release.sh")
    health = release.split("service_healthy", 1)[1].split("wait_for_service", 1)[0]

    assert 'local expected_sha="${1:-}"' in health
    assert "effective_sha" in health
    assert "broker_auth_dependency_degraded" in health
    assert "unknown startup failures still trigger rollback" in health
    assert 'wait_for_service "$AFTER"' in release
    assert 'service_healthy_confirmed "$BEFORE"' in release
    assert "deployed_dependency_degraded" in release


def _market_session_release_helpers() -> str:
    """Execute only the pure time-gate functions, never the deployment body."""
    release = _text("deploy/lightsail_release.sh")
    assert release.count("market_session_protected() {") == 1
    assert release.count("auto_release_should_defer() {") == 1
    return (
        "market_session_protected() {"
        + release.split("market_session_protected() {", 1)[1].split("\nexec 9>", 1)[0]
    )


def test_auto_release_market_window_boundary_and_failure_modes() -> None:
    helpers = _market_session_release_helpers()
    cases = [
        ("1", "0759", False),
        ("1", "0800", True),
        ("1", "0844", True),
        ("1", "0845", True),
        ("5", "1230", True),
        ("5", "1545", True),
        ("5", "1546", False),
        ("6", "1230", False),
        ("7", "1230", False),
        ("", "", True),
        ("5", "2580", True),
    ]
    for weekday, hhmm, defer in cases:
        result = subprocess.run(
            [
                "bash",
                "-c",
                helpers + '\nmarket_session_protected "$1" "$2"',
                "fixture",
                weekday,
                hhmm,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (result.returncode == 0) is defer, (weekday, hhmm, result.stderr)


def test_auto_release_only_and_manual_emergency_bypass() -> None:
    helpers = _market_session_release_helpers()
    shell = helpers + """
AUTO_MODE="$1"
FAKE_DAY="$2"
FAKE_CLOCK="$3"
date() {
  case "$1" in
    +%u) printf '%s' "$FAKE_DAY";;
    +%H%M) printf '%s' "$FAKE_CLOCK";;
    *) return 1;;
  esac
}
auto_release_should_defer
"""
    for mode, weekday, hhmm, defer in (
        ("true", "5", "1225", True),
        ("true", "6", "1225", False),
        ("false", "5", "1225", False),
    ):
        result = subprocess.run(
            ["bash", "-c", shell, "fixture", mode, weekday, hhmm],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (result.returncode == 0) is defer, (mode, weekday, result.stderr)


def test_auto_market_gate_preserves_health_recovery_and_research_poll() -> None:
    release = _text("deploy/lightsail_release.sh")
    gate = release.index("if auto_release_should_defer; then")
    protected = release[gate : release.index("\nchmod +x", gate)]
    assert release.index("exec 9>") < release.index('cd "$APP_DIR"') < gate
    assert release.index("git fetch --quiet origin main") < gate
    assert "service_healthy_confirmed" in protected
    assert 'sudo -n systemctl restart "$SERVICE"' in protected
    assert "runtime_recovery_failed" in protected
    assert "runtime_checkout_mismatch" in protected
    assert 'current_runtime_sha" != "$BEFORE"' in protected
    assert '"http://127.0.0.1:${PORT}/releasez"' in protected
    assert protected.index("runtime_checkout_mismatch") < protected.index(
        'sudo -n systemctl restart "$SERVICE"'
    )
    assert "unvalidated revision" in protected
    assert "poll_research_request" in protected
    assert 'if [ "$BEFORE" != "$AFTER" ]; then' in protected
    assert "write_status deferred_market_window" in protected
    assert "write_status current" in protected
    assert gate < release.index("\nvalidate_environment\n", gate)
    assert "TZ=Asia/Kolkata date +%u" in release
    assert "TZ=Asia/Kolkata date +%H%M" in release


def test_auto_market_guard_rechecks_before_checkout_and_restart() -> None:
    release = _text("deploy/lightsail_release.sh")
    validated = release.index('write_status validating "validating')
    after_candidate = release.index("if auto_release_should_defer; then", validated)
    checkout = release.index('git reset --hard --quiet "$AFTER"', after_candidate)
    restart_gate = release.index("if auto_release_should_defer; then", checkout)
    engine_restart = release.index('sudo systemctl restart "$SERVICE"', restart_gate)
    assert validated < after_candidate < checkout < restart_gate < engine_restart
    assert "checkout untouched" in release[after_candidate:checkout]
    assert 'set_runtime_build_sha "$BEFORE"' in release[restart_gate:engine_restart]
    assert 'git reset --hard --quiet "$BEFORE"' in release[restart_gate:engine_restart]
    assert (
        subprocess.run(
            ["bash", "-n", str(ROOT / "deploy/lightsail_release.sh")],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )


def test_forced_restart_survives_market_boundary_deferral() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert (
        'RESTART_PENDING_FILE="${CONFIG_DIR}/auto_release_restart_pending"'
        in release
    )
    assert 'if [ -f "$RESTART_PENDING_FILE" ]; then FORCE_RESTART=true; fi' in release
    assert release.count(': > "$RESTART_PENDING_FILE"') == 2
    assert 'rm -f "$RESTART_PENDING_FILE"' in release
    assert release.index('rm -f "$RESTART_PENDING_FILE"') > release.index(
        'if wait_for_service "$AFTER"; then'
    )
