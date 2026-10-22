"""Architecture checks for the AWS Lightsail release and secret boundary."""

from __future__ import annotations

import os
import re
import subprocess
import sys
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


def test_sudoers_policy_authorizes_exact_automation_controls() -> None:
    setup = _text("deploy/lightsail_setup.sh")
    release = _text("deploy/lightsail_release.sh")

    assert 'if [ "${1:-}" = "--repair-control-sudoers" ]; then' in setup
    assert "sudo -n visudo -cf" in setup
    assert "sudo -n install -m 0440" in setup
    assert "ubuntu ALL=(root) NOPASSWD:" in setup
    for service in (
        "niftybot",
        "niftybot-admin",
        "niftybot-streamlit",
        "niftybot-autodeploy.service",
    ):
        assert service in setup
    assert 'rules+=("$executable restart --no-block $service")' in setup
    assert "--repair-control-sudoers" in release
    assert "control_sudoers_v2_installed" in release
    repair = release.index("--repair-control-sudoers")
    early_gate = release.index("if auto_release_should_defer; then")
    assert repair < early_gate
    assert (
        subprocess.run(
            ["bash", "-n", str(ROOT / "deploy/lightsail_setup.sh")],
            check=False,
        ).returncode
        == 0
    )


def test_legacy_installer_never_executes_full_setup_as_sudoers_repair(
    tmp_path,
) -> None:
    """Legacy scripts silently ignored unknown flags and ran full provisioning."""
    release = _text("deploy/lightsail_release.sh")
    start = release.index("control_sudoers_repair_supported() {")
    end = release.index("\n# Repair on a current checkout", start)
    helpers = release[start:end]
    legacy = tmp_path / "legacy-setup.sh"
    legacy.write_text(
        """#!/usr/bin/env bash
printf BAD > "$SENTINEL"
""",
        encoding="utf-8",
    )
    safe = tmp_path / "current-setup.sh"
    safe.write_text(
        """#!/usr/bin/env bash
if [ "${1:-}" = "--repair-control-sudoers" ]; then
  printf REPAIR_ONLY > "$SENTINEL"
  exit 0
fi
exit 5
""",
        encoding="utf-8",
    )
    import os

    sentinel = tmp_path / "actions.log"
    env = dict(os.environ, SENTINEL=str(sentinel))
    cmd = helpers + """
CONFIG_DIR="$3"
log() { :; }
repair_control_sudoers_from "$1"
"""
    for installer, expected in (
        (legacy, False),
        (safe, True),
    ):
        sentinel.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", "-c", cmd, "fixture", str(installer), "ignored", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert (result.returncode == 0) is expected
        if expected:
            assert sentinel.read_text() == "REPAIR_ONLY"
            assert (tmp_path / "control_sudoers_v2_installed").exists()
        else:
            assert not sentinel.exists()


def test_repair_only_uses_validated_candidate_before_checkout_mutation() -> None:
    release = _text("deploy/lightsail_release.sh")
    validated = release.index('write_status validating "validating')
    late_repair = release.index(
        'repair_control_sudoers_from "$CANDIDATE/deploy/lightsail_setup.sh"',
        validated,
    )
    checkout = release.index('git reset --hard --quiet "$AFTER"', late_repair)
    guard = release.index("if auto_release_should_defer; then", validated)
    assert validated < guard < late_repair < checkout
    assert "control_permissions_failed" in release
    assert "market_session_protected()" not in release


def test_retired_dashboard_and_adhoc_tooling_are_absent() -> None:
    retired = (
        "dashboard/operations_console.py",
        "scripts/assess_all.sh",
        "scripts/run_sync.sh",
        "mypy.ini",
        "config/defaults.yaml",
        "config/events.yaml",
        "config/order_execution.yaml",
        "config/strategy.yaml",
    )
    assert [path for path in retired if (ROOT / path).exists()] == []
    assert (ROOT / "dashboard/superlite_console.py").is_file()
    assert (ROOT / "dashboard/streamlit_app.py").is_file()
    assert (ROOT / "dashboard/log_export.py").is_file()


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
    assert '[ ! -f "$CANDIDATE/dashboard/superlite_console.py" ]' in release
    assert '"$VENV/bin/python" -m compileall -q "$CANDIDATE/dashboard"' in release
    assert "required dashboard entrypoint missing" in release
    assert '"$VENV/bin/python" -m py_compile' not in release
    assert '"$CANDIDATE/dashboard/event_buffer.py"' not in release
    assert "tests/dashboard/test_superlite_event_details.py" in release
    assert not (ROOT / "dashboard/event_buffer.py").exists()
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


def test_dashboard_candidate_compilation_survives_retired_helper_removal(
    tmp_path: Path,
) -> None:
    """The release must not demand a removed, hard-coded helper module."""
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    (dashboard / "superlite_console.py").write_text(
        "from superlite_events import parse_event\n",
        encoding="utf-8",
    )
    (dashboard / "superlite_events.py").write_text(
        "def parse_event(value): return value\n",
        encoding="utf-8",
    )
    assert not (dashboard / "event_buffer.py").exists()
    compile_cmd = [sys.executable, "-m", "compileall", "-q", str(dashboard)]
    success = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    assert success.returncode == 0, success.stderr

    # A newly introduced syntax error must still block the validated release.
    (dashboard / "new_dashboard_helper.py").write_text(
        "def new_helper(:\n", encoding="utf-8"
    )
    failure = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    assert failure.returncode != 0
    assert "SyntaxError" in (failure.stdout + failure.stderr)


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


def test_admin_recovery_is_independent_of_broker_and_ci_gates() -> None:
    release = _text("deploy/lightsail_release.sh")
    entry = release.index("exec 9>")
    recovery_call = release.index("recover_admin_control || true", entry)
    broker_gate = release.index("if auto_release_should_defer; then", entry)
    ci_gate = release.index('if [ "$BEFORE" != "$AFTER" ]; then', broker_gate)
    assert entry < recovery_call < broker_gate < ci_gate
    assert "http://127.0.0.1:${ADMIN_PORT}/healthz" in release
    assert 'http://127.0.0.1:${ADMIN_PORT}/admin"' in release
    recovery = release.split("recover_admin_control() {", 1)[1].split(
        "\nrestart_streamlit()", 1
    )[0]
    assert "restart_admin && admin_healthy" in recovery
    assert 'systemctl restart "$SERVICE"' not in recovery
    assert "intraday_flat_broker_confirmed" not in recovery


def test_isolated_admin_recovery_success_and_failure(tmp_path: Path) -> None:
    release = _text("deploy/lightsail_release.sh")
    recovery = (
        "recover_admin_control() {"
        + release.split("recover_admin_control() {", 1)[1].split(
            "\nrestart_streamlit()", 1
        )[0]
    )
    shell = recovery + """
log() { printf "%s\\n" "$*" >> "$LOGFILE"; }
admin_healthy() { [ "$ADMIN_STATE" = "healthy" ]; }
restart_admin() {
  printf "restart-admin\\n" >> "$LOGFILE"
  if [ "$RESTART_SUCCEEDS" = "true" ]; then ADMIN_STATE=healthy; return 0; fi
  return 1
}
recover_admin_control
"""
    for initial, restarted, expected in (
        ("healthy", "false", 0),
        ("unhealthy", "true", 0),
        ("unhealthy", "false", 1),
    ):
        logfile = tmp_path / "admin.log"
        logfile.unlink(missing_ok=True)
        env = dict(
            os.environ,
            LOGFILE=str(logfile),
            ADMIN_STATE=initial,
            RESTART_SUCCEEDS=restarted,
            ADMIN_SERVICE="niftybot-admin",
        )
        result = subprocess.run(
            ["bash", "-c", shell], capture_output=True, text=True, env=env, check=False
        )
        assert result.returncode == expected, result.stderr
        logged = logfile.read_text() if logfile.exists() else ""
        assert logged.count("restart-admin") == (0 if initial == "healthy" else 1)
        assert "trading engine unchanged" in logged or initial == "healthy"


def test_changed_release_restarts_current_revision_admin_before_research_poll() -> None:
    release = _text("deploy/lightsail_release.sh")

    assert 'ADMIN_SERVICE="${BOT_ADMIN_SERVICE_NAME:-niftybot-admin}"' in release
    assert "restart_admin()" in release
    helper = release.split("restart_admin()", 1)[1].split("restart_streamlit()", 1)[0]
    assert 'systemctl restart "$ADMIN_SERVICE"' in helper
    assert 'http://127.0.0.1:${ADMIN_PORT}/healthz"' in helper
    assert 'http://127.0.0.1:${ADMIN_PORT}/admin"' in helper
    assert "http://127.0.0.1:${ADMIN_PORT}/admin/api/status" not in helper
    assert "--max-time 2" in helper
    assert "--max-time 3" in helper

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


def test_canonical_deployment_scripts_do_not_embed_credentials() -> None:
    paths = (
        "deploy/lightsail_setup.sh",
        "deploy/lightsail_release.sh",
    )
    assignment = re.compile(
        r"(?m)^(?:KITE|ZERODHA|BROKER)_(?:API_KEY|API_SECRET|ACCESS_TOKEN)=([^\s]*)$"
    )
    for path in paths:
        text = _text(path)
        for match in assignment.finditer(text):
            assert match.group(1) == "", f"credential literal found in {path}"


def test_canonical_redeploy_path_replaces_two_unused_force_aliases() -> None:
    assert not (ROOT / "deploy/scripts/update_and_deploy.sh").exists()
    assert not (ROOT / "ops/redeploy.sh").exists()

    setup = _text("deploy/lightsail_setup.sh")
    release = _text("deploy/lightsail_release.sh")
    guide = _text("deploy/LIGHTSAIL_SETUP.md")

    # Automatic and documented manual operations both invoke the one
    # validated release owner; there is no bypassing forwarding wrapper.
    assert "deploy/lightsail_release.sh --auto" in setup
    assert "bash deploy/lightsail_release.sh --force" in guide
    assert "--force|--force-restart) FORCE_RESTART=true" in release
    assert "git fetch --quiet origin main" in release


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


def test_auto_release_never_defers_on_engine_arming_or_broker_exposure() -> None:
    """Code releases must not be gated on a possibly always-armed runner."""
    release = _text("deploy/lightsail_release.sh")
    assert "auto_release_should_defer" not in release
    assert "intraday_flat_broker_confirmed" not in release
    assert "deferred_broker_exposure" not in release
    assert "deferred_unsafe_recovery" not in release
    assert "PY_FLAT_GUARD" not in release
    assert "BOT_RELEASE_KITE_ACCESS_TOKEN" not in release


def test_release_uses_single_fetch_and_same_auto_manual_validation() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert release.count("git fetch --quiet origin main") == 1
    assert release.index("git fetch --quiet origin main") < release.index(
        'write_status validating "validating'
    )
    assert release.index('write_status validating "validating') < release.index(
        'git reset --hard --quiet "$AFTER"'
    )
    assert release.index('set_runtime_build_sha "$AFTER"') < release.index(
        'sudo systemctl restart "$SERVICE"'
    )
    assert 'write_status deferred_ci' in release
    assert 'write_status validation_failed' in release
    assert 'write_status rolled_back' in release


def test_pending_restart_marker_consumed_after_successful_release() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert "auto_release_restart_pending" in release
    assert 'if [ -f "$RESTART_PENDING_FILE" ]; then FORCE_RESTART=true; fi' in release
    assert 'rm -f "$RESTART_PENDING_FILE"' in release
    assert release.index('rm -f "$RESTART_PENDING_FILE"') > release.index(
        'if wait_for_service "$AFTER"; then'
    )


def test_immediate_release_script_passes_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "deploy/lightsail_release.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_successful_engine_release_never_hides_failed_admin(tmp_path: Path) -> None:
    """A healthy engine must not mask a broken independent admin service."""
    release = _text("deploy/lightsail_release.sh")
    start = release.index('if wait_for_service "$AFTER"; then')
    end = release.index('\nif [ "$BEFORE" != "$AFTER" ]; then\n  git reset', start)
    success_block = release[start:end]

    script = (
        "wait_for_service() { return 0; }\n"
        'restart_admin() { [ "$ADMIN_OK" = "yes" ]; }\n'
        'restart_streamlit() { [ "$CONSOLE_OK" = "yes" ]; }\n'
        'broker_auth_dependency_degraded() { [ "$BROKER_DEGRADED" = "yes" ]; }\n'
        'write_status() { printf "%s\\n" "$1" > "$STATUS_OUTPUT"; }\n'
        "logger() { :; }\n"
        "log() { :; }\n"
        "poll_research_request() { :; }\n" + success_block
    )
    cases = (
        ("yes", "yes", "no", "deployed"),
        ("no", "yes", "no", "deployed_admin_degraded"),
        ("no", "no", "no", "deployed_controls_degraded"),
        ("yes", "yes", "yes", "deployed_dependency_degraded"),
        ("no", "yes", "yes", "deployed_admin_and_broker_degraded"),
    )
    for admin_ok, console_ok, broker_degraded, expected in cases:
        state_file = tmp_path / "state.txt"
        result = subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "ADMIN_OK": admin_ok,
                "CONSOLE_OK": console_ok,
                "BROKER_DEGRADED": broker_degraded,
                "STATUS_OUTPUT": str(state_file),
                "RESTART_PENDING_FILE": str(tmp_path / "pending"),
                "AFTER": "f" * 40,
                "BEFORE": "e" * 40,
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert state_file.read_text(encoding="utf-8").strip() == expected


def test_current_release_status_surfaces_failed_admin_recovery() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert release.count("write_status current_admin_degraded") == 2
    assert release.count("if admin_healthy; then") >= 2
    assert "admin service unavailable after recovery attempt" in release


def test_admin_status_link_is_valid_on_engine_and_isolated_admin() -> None:
    page = _text("src/nifty_scalper_bot/admin_dashboard.py")
    assert 'href="/admin/status.json">Service status</a>' in page
    assert 'href="/health">Health</a>' not in page
    assert '@router.get("/admin/status.json")' in page
