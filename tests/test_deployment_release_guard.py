from __future__ import annotations

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
