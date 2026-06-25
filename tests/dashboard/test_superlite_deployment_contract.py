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
