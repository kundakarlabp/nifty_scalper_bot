"""Test that automated releases retain only bounded, private env snapshots."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = ROOT / "deploy" / "lightsail_release.sh"


def _run_backup(
    tmp_path: Path, *, source_exists: bool = True
) -> tuple[subprocess.CompletedProcess[str], Path]:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    env_file = config_dir / "niftybot.env"
    if source_exists:
        env_file.write_text("KITE_API_SECRET=private-example\n", encoding="utf-8")
        env_file.chmod(0o600)

    script = RELEASE_SCRIPT.read_text(encoding="utf-8")
    begin = script.index("save_protected_env_backup() {")
    end = script.index("\n}", begin) + 2
    function = script[begin:end]

    result = subprocess.run(
        [
            "bash",
            "-e",
            "-u",
            "-o",
            "pipefail",
            "-c",
            function + "\n"+"save_protected_env_backup",
        ],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "CONFIG_DIR": str(config_dir), "ENV_FILE": str(env_file)},
    )
    return result, config_dir


def test_automatic_env_backup_retention_is_bounded_and_redacted(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    backups = config_dir / "backups"
    backups.mkdir(parents=True)
    for i in range(1, 10):
        item = backups / f"niftybot.env.202610{i:02d}_100000"
        item.write_text("KITE_API_SECRET=older-private-value\n", encoding="utf-8")
        item.chmod(0o600)
    manual = backups / "manual-operator-export.env"
    manual.write_text("manual-preserve", encoding="utf-8")
    outside = config_dir / "unchanged-secret"
    outside.write_text("do-not-touch", encoding="utf-8")
    symlink = backups / "niftybot.env.20260101_000000"
    symlink.symlink_to(outside)
    anomalous = backups / "niftybot.env.not-a-release"
    anomalous.write_text("operator-kept", encoding="utf-8")

    result, config_dir = _run_backup(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "private-" not in result.stdout + result.stderr
    assert (config_dir / "niftybot.env").read_text(encoding="utf-8") == (
        "KITE_API_SECRET=private-example\n"
    )
    assert manual.exists() and anomalous.exists()
    assert symlink.is_symlink()
    assert outside.read_text(encoding="utf-8") == "do-not-touch"
    managed = [
        p for p in backups.iterdir()
        if p.is_file() and not p.is_symlink() and p.name.startswith("niftybot.env.")
        and p.name != "niftybot.env.not-a-release"
    ]
    assert len(managed) == 5
    assert any("20261009" in p.name for p in managed)
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in managed)
    assert stat.S_IMODE(backups.stat().st_mode) == 0o700


def test_backup_failure_does_not_prune_existing_recoverable_files(
    tmp_path: Path,
) -> None:
    backups = tmp_path / "config" / "backups"
    backups.mkdir(parents=True)
    previous = backups / "niftybot.env.20261024_100000"
    previous.write_text("keep-for-rollback", encoding="utf-8")

    result, _ = _run_backup(tmp_path, source_exists=False)

    assert result.returncode != 0
    assert previous.read_text(encoding="utf-8") == "keep-for-rollback"
    assert len(list(backups.iterdir())) == 1


def test_release_calls_backup_before_applying_candidate() -> None:
    script = RELEASE_SCRIPT.read_text(encoding="utf-8")
    assert "MAX_MANAGED_ENV_BACKUPS=5" in script
    assert 'if ! save_protected_env_backup; then' in script
    assert script.index("if ! save_protected_env_backup; then") < script.index(
        'write_status deploying "deploying'
    )
    assert 'write_status backup_failed' in script
