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
    assert "ExecStart=/usr/bin/env bash $APP_DIR/deploy/lightsail_release.sh --auto" in setup


def test_release_runner_validates_and_rolls_back() -> None:
    release = _text("deploy/lightsail_release.sh")
    assert 'flock -n 9' in release
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
    assert 'http://127.0.0.1:${PORT}/livez' in release
    health_block = release.split("service_healthy", 1)[1].split("wait_for_service", 1)[0]
    assert "/readyz" not in health_block
    assert 'git reset --hard --quiet "$BEFORE"' in release
    assert 'sudo systemctl restart "$SERVICE"' in release


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
    assert 'AUTODEPLOY_SERVICE="${BOT_AUTODEPLOY_SERVICE_NAME:-niftybot-autodeploy}"' in release
    migration_block = release.split("migrate_autodeploy_entrypoint", 1)[1].split(
        "restart_streamlit", 1
    )[0]
    assert "ExecStart=/usr/bin/env bash ${APP_DIR}/deploy/lightsail_release.sh --auto" in migration_block
    assert "AUTODEPLOY_ENTRYPOINT_MIGRATED=true" in migration_block
    assert "sudo systemctl daemon-reload" in migration_block


def test_lightsail_migration_forces_restart_before_healthy_no_change_exit() -> None:
    release = _text("deploy/lightsail_release.sh")
    migration_call = release.index("migrate_systemd_entrypoint")
    force_restart = release.index('FORCE_RESTART=true', migration_call)
    healthy_no_change = release.index('if [ "$BEFORE" = "$AFTER" ]', force_restart)
    assert migration_call < force_restart < healthy_no_change
    candidate_index = release.index("CANDIDATE=", healthy_no_change)
    no_change_block = release[healthy_no_change:candidate_index]
    assert '[ "$FORCE_RESTART" = false ]' in no_change_block


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
