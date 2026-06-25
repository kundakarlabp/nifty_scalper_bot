from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_combined_ops_unit_is_bounded_and_independent() -> None:
    unit = (ROOT / "deploy/systemd/niftybot-streamlit.service").read_text(encoding="utf-8")
    assert "nifty_scalper_bot.superlite_admin:app" in unit
    assert "dashboard/superlite_console.py" in unit
    assert "--no-access-log" in unit
    assert "--server.fileWatcherType=none" in unit
    assert "MemoryMax=420M" in unit


def test_installer_preserves_external_environment() -> None:
    installer = (ROOT / "deploy/scripts/install_streamlit_console.sh").read_text(encoding="utf-8")
    assert "touch \"$ENV_FILE\"" in installer
    assert "ensure_default POST_MARKET_QUIET_MODE true" in installer
    assert "enable --quiet --now niftybot-autodeploy.timer" in installer
    assert "niftybot-dashboard-update" in installer
    assert "rm -rf -- {}" in installer
