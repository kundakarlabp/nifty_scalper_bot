"""Keep the legacy CLI on one private env file, without broker calls."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
from scripts.zerodha_token_cli import resolve_token_env_path, write_env_tokens


def test_production_uses_canonical_env_only(tmp_path: Path) -> None:
    protected = tmp_path / "niftybot.env"
    protected.write_text("ENABLE_LIVE=false\n", encoding="utf-8")
    environment = {
        "DEPLOYMENT_PLATFORM": "aws_lightsail",
        "BOT_ENV_FILE": str(protected),
    }
    assert resolve_token_env_path(environ=environment) == protected
    with pytest.raises(ValueError, match="canonical"):
        resolve_token_env_path(str(tmp_path / ".env"), environ=environment)


@pytest.mark.parametrize(
    "environment",
    [
        {"DEPLOYMENT_PLATFORM": "aws_lightsail"},
        {"DEPLOYMENT_PLATFORM": "aws_lightsail", "BOT_ENV_FILE": ".env"},
    ],
)
def test_production_rejects_missing_or_relative_operator_path(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        resolve_token_env_path(environ=environment)


def test_production_rejects_missing_or_symlinked_operator_file(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.env"
    real.write_text("ENABLE_LIVE=false\n", encoding="utf-8")
    link = tmp_path / "link.env"
    link.symlink_to(real)
    for candidate in (tmp_path / "absent.env", link):
        with pytest.raises(ValueError):
            resolve_token_env_path(
                environ={
                    "DEPLOYMENT_PLATFORM": "aws_lightsail",
                    "BOT_ENV_FILE": str(candidate),
                }
            )


def test_development_can_still_use_local_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_token_env_path(environ={}) == Path(".env")


def test_atomic_private_token_update_preserves_unrelated_settings(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "operator.env"
    path.write_text(
        "ENABLE_LIVE=false\n"
        "EXECUTION_MODE=SHADOW\n"
        "KITE_ACCESS_TOKEN=stale\n"
        "export BROKER_ACCESS_TOKEN=old\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    token = "sample-test-token"
    write_env_tokens(path, token)
    data = path.read_text(encoding="utf-8")
    assert "ENABLE_LIVE=false\nEXECUTION_MODE=SHADOW" in data
    assert all(
        data.count(f"{name}={token}") == 1
        for name in ("ZERODHA_ACCESS_TOKEN", "BROKER_ACCESS_TOKEN", "KITE_ACCESS_TOKEN")
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".nifty-token-*"))
    write_env_tokens(path, token)
    assert all(
        path.read_text(encoding="utf-8").count(f"{name}={token}") == 1
        for name in ("ZERODHA_ACCESS_TOKEN", "BROKER_ACCESS_TOKEN", "KITE_ACCESS_TOKEN")
    )


def test_token_writer_rejects_symlink_and_injected_lines(tmp_path: Path) -> None:
    real = tmp_path / "secrets.env"
    real.write_text("ENABLE_LIVE=false\n", encoding="utf-8")
    link = tmp_path / "link.env"
    link.symlink_to(real)
    with pytest.raises(ValueError):
        write_env_tokens(link, "fake-token")
    with pytest.raises(ValueError):
        write_env_tokens(real, "fake-token\nENABLE_LIVE=true")
    assert real.read_text(encoding="utf-8") == "ENABLE_LIVE=false\n"


def test_manage_bot_shell_cannot_source_repo_env() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "manage_bot.sh").read_text(encoding="utf-8")
    assert "source .env" not in text
    assert "exec python scripts/zerodha_token_cli.py" in text
    assert '--env "$BOT_ENV_FILE" --write-env' in text
    assert "absolute BOT_ENV_FILE" in text


def test_unconfigured_manual_shell_does_not_run_token_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "manage_bot.sh"), "token"],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=8,
    )
    assert result.returncode == 2
    assert "absolute BOT_ENV_FILE" in result.stderr
