from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_docker_image_embeds_and_runs_verified_release() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "ARG RAILWAY_GIT_COMMIT_SHA=unknown" in dockerfile
    assert "/app/.build_commit_sha" in dockerfile
    assert "ENV APP_BUILD_SHA=${RAILWAY_GIT_COMMIT_SHA}" in dockerfile
    assert "nifty_scalper_bot.deployment_main:app" in dockerfile
    assert "http://localhost:${PORT:-8080}/releasez" in dockerfile
    assert "--port ${PORT:-8080}" in dockerfile
    assert "nifty_scalper_bot.main:app" not in dockerfile


def test_release_verification_precedes_trading_app_import() -> None:
    source = Path("src/nifty_scalper_bot/deployment_main.py").read_text(
        encoding="utf-8"
    )
    enforce_at = source.index("_RELEASE = enforce_release_freshness()")
    import_at = source.index("from nifty_scalper_bot.main import app")
    watchdog_at = source.index("start_release_watchdog_thread(_RELEASE)")
    assert enforce_at < watchdog_at < import_at
    assert '@app.get("/releasez")' in source


def test_railway_activates_only_release_verified_instance() -> None:
    railway = Path("railway.toml").read_text(encoding="utf-8")
    assert 'healthcheckPath = "/releasez"' in railway
    assert "healthcheckTimeout = 300" in railway
    assert "overlapSeconds = 0" in railway
    assert "drainingSeconds = 30" in railway
    assert 'restartPolicyType = "ON_FAILURE"' in railway
    assert "restartPolicyMaxRetries = 10" in railway


def test_redundant_double_deploy_workflow_is_absent() -> None:
    assert not Path(".github/workflows/railway-redeploy.yml").exists()


def test_compose_and_dockerfile_use_canonical_release_liveness() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "http://localhost:8080/releasez" in compose
    assert "http://localhost:8080/readyz" not in compose
    assert "http://localhost:${PORT:-8080}/releasez" in dockerfile
    assert "nifty_scalper_bot.deployment_main:app" in dockerfile
    assert "${PORT:-8080}" in dockerfile
    assert '"--port", "8080"' in compose
    assert "nifty_scalper_bot.app" not in compose
    assert not Path("deploy/scripts/entrypoint.sh").exists()


def test_retired_entrypoints_and_standalone_helpers_stay_absent() -> None:
    retired = (
        "deploy/install.sh",
        "deploy/setup_swap.sh",
        "deploy/systemd/nifty-admin.service",
        "deploy/systemd/nifty-scalper.service",
        "scripts/start_bot_morning.sh",
        "scripts/stop_bot_evening.sh",
        "deploy/scripts/entrypoint.sh",
        "config/__init__.py",
        "data/__init__.py",
        "tools/normalize_subscription_symbols.py",
        "tools/async_await_audit.py",
        "tools/nightly_replay.py",
        "src/nifty_scalper_bot/admin_ops_dashboard.py",
    )
    assert [path for path in retired if Path(path).exists()] == []
    assert Path("deploy/lightsail_setup.sh").is_file()
    assert Path("deploy/lightsail_release.sh").is_file()
    assert Path("deploy/systemd/niftybot-admin.service").is_file()
    assert Path("src/nifty_scalper_bot/superlite_admin.py").is_file()
    assert Path("dashboard/superlite_console.py").is_file()


def test_obsolete_health_and_real_order_smoke_scripts_stay_retired() -> None:
    retired = (
        "scripts/healthcheck.py",
        "scripts/run_health_server.py",
        "scripts/verify_runtime.py",
        "tools/amo_smoke.py",
    )
    assert not [path for path in retired if Path(path).exists()]
    main = Path("src/nifty_scalper_bot/main.py").read_text(encoding="utf-8")
    assert '@app.get("/livez")' in main
    assert '@app.get("/health/trading")' in main
    assert '@app.get("/trading/status")' in main
    assert Path("scripts/test_balance_fetch.py").is_file()
    assert Path("scripts/ws_smoke_test.py").is_file()


def test_backup_runs_without_root_config_and_does_not_capture_env(
    tmp_path: Path,
) -> None:
    script = Path("scripts/backup_data.sh")
    assert (
        subprocess.run(
            ["bash", "-n", str(script)], check=False, capture_output=True
        ).returncode
        == 0
    )
    assert "backups/" in Path(".gitignore").read_text().splitlines()
    assert "backups/" in Path(".dockerignore").read_text().splitlines()

    root = tmp_path / "sandbox"
    (root / "scripts").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "data").mkdir()
    (root / "scripts/backup_data.sh").write_text(script.read_text())
    (root / "logs/bot.log").write_text("healthy")
    (root / "data/state.json").write_text("{}")
    (root / ".env").write_text("FAKE_SECRET=do-not-copy")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_rsync = bindir / "rsync"
    fake_rsync.write_text('#!/bin/sh\nset -eu\nmkdir -p "$4"\ncp -R "$3/." "$4/"\n')
    fake_rsync.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        ["bash", str(root / "scripts/backup_data.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outputs = list((root / "backups").iterdir())
    assert len(outputs) == 1
    backup = outputs[0]
    assert (backup / "logs/bot.log").read_text() == "healthy"
    assert (backup / "data/state.json").read_text() == "{}"
    assert not (backup / ".env").exists()
    assert not (backup / "config").exists()
