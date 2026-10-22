"""Regression contracts for an automatic non-overlapping release timer."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_new_installs_use_30_second_post_run_checks() -> None:
    setup = (ROOT / "deploy/lightsail_setup.sh").read_text(encoding="utf-8")
    assert "OnBootSec=30s" in setup
    assert "OnUnitInactiveSec=30s" in setup
    assert "AccuracySec=5s" in setup
    assert "OnUnitActiveSec=2min" not in setup


def test_existing_host_timer_upgrades_without_reinstalling() -> None:
    release = (ROOT / "deploy/lightsail_release.sh").read_text(encoding="utf-8")
    assert release.count("configure_autodeploy_timer() {") == 1
    assert release.count("\nconfigure_autodeploy_timer\n") == 1
    assert 'grep -Fqx "OnUnitInactiveSec=30s"' in release
    assert "sudo -n python3 -" in release
    assert "sudo -n systemctl daemon-reload" in release
    assert "WARNING: timer upgrade needs host permissions" in release
    call = release.index("configure_autodeploy_timer\nif")
    assert call < release.index("git fetch --quiet origin main", call)
