from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_admin_and_review_services_are_independent_and_bounded() -> None:
    admin = (ROOT / "deploy/systemd/niftybot-admin.service").read_text(encoding="utf-8")
    review = (ROOT / "deploy/systemd/niftybot-streamlit.service").read_text(encoding="utf-8")

    assert "nifty_scalper_bot.superlite_admin:app" in admin
    assert "--no-access-log" in admin
    assert "MemoryMax=180M" in admin
    assert "dashboard/superlite_console.py" in review
    assert "--server.fileWatcherType=none" in review
    assert "NoNewPrivileges=true" in review
    assert "MemoryMax=320M" in review


def test_installer_preserves_external_environment() -> None:
    installer = (ROOT / "deploy/scripts/install_streamlit_console.sh").read_text(encoding="utf-8")
    assert "touch \"$ENV_FILE\"" in installer
    assert "ensure_default POST_MARKET_QUIET_MODE true" in installer
    assert "enable --quiet --now niftybot-autodeploy.timer" in installer
    assert "niftybot-admin.service niftybot-streamlit.service" in installer
    assert "rm -rf -- {}" in installer


def test_update_control_delegates_only_to_validated_autodeployer() -> None:
    admin = (ROOT.parent / "src/nifty_scalper_bot/admin_dashboard.py").read_text(
        encoding="utf-8"
    )
    superlite = (ROOT.parent / "src/nifty_scalper_bot/superlite_admin.py").read_text(
        encoding="utf-8"
    )
    service_control = (
        ROOT.parent / "src/nifty_scalper_bot/ops/service_control.py"
    ).read_text(encoding="utf-8")

    update_block = admin.split("def _git_update()", 1)[1].split(
        "# ---------------- UI ----------------", 1
    )[0]
    assert "restart_deployer()" in update_block
    assert "git" not in update_block
    assert "pull" not in update_block

    assert "restart_deployer()" in superlite
    assert "niftybot-autodeploy.service" in service_control
    assert 'action="restart_deployer"' in service_control
