"""Ensure installed Lightsail deployments adopt the low-overhead fast release timer."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_new_installs_use_post_run_timer_not_overlap_prone_active_interval() -> None:
    setup = (ROOT / "deploy/lightsail_setup.sh").read_text(encoding="utf-8")
    assert "OnBootSec=30s" in setup
    assert "OnUnitInactiveSec=30s" in setup
    assert "AccuracySec=5s" in setup
    assert "OnUnitActiveSec=2min" not in setup


def test_existing_host_timer_is_migrated_without_full_setup() -> None:
    release = (ROOT / "deploy/lightsail_release.sh").read_text(
        encoding="utf-8"
    )
    assert release.count("configure_autodeploy_timer() {") == 1
    assert release.count("\nconfigure_autodeploy_timer\n") == 1
    assert "sudo -n systemctl daemon-reload" in release
    assert 'sudo -n systemctl restart "${AUTODEPLOY_SERVICE}.timer"' in release
    assert 'grep -Fqx "OnUnitInactiveSec=30s"' in release
    assert "sudo -n python3 -" in release
    assert "WARNING: timer upgrade needs host permissions" in release
    assert release.index("configure_autodeploy_timer\nif") < release.index(
        "git fetch --quiet origin main", release.index("configure_autodeploy_timer\nif")
    )


def test_timer_migration_keeps_release_script_valid_bash() -> None:
    release = ROOT / "deploy/lightsail_release.sh"
    result = subprocess.run(
        ["bash", "-n", str(release)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
