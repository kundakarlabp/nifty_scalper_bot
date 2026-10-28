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


class _RecoveredKite:
    def __init__(self, api_key: str, **_kwargs):
        assert api_key == "production-key"
        self.token = None

    def set_access_token(self, token: str) -> None:
        self.token = token

    def profile(self):
        return {"user_id": "AB1234"}

    def margins(self, segment: str):
        assert segment == "equity"
        return {"available": {"cash": 0.0, "live_balance": 5000.0}}


def _configure_saved_session(monkeypatch, tmp_path: Path):
    from scripts import zerodha_token_cli as cli

    env_file = tmp_path / "niftybot.env"
    env_file.write_text(
        "BROKER_API_KEY=production-key\n"
        "BROKER_API_SECRET=production-secret\n"
        "BROKER_ACCESS_TOKEN=saved-token\n"
        "ADMIN_ZERODHA_USER_ID=AB1234\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("BOT_ENV_FILE", str(env_file))
    monkeypatch.setenv("BROKER_API_KEY", "production-key")
    monkeypatch.setenv("BROKER_API_SECRET", "production-secret")
    monkeypatch.setenv("BROKER_ACCESS_TOKEN", "saved-token")
    monkeypatch.setenv("ADMIN_ZERODHA_USER_ID", "AB1234")
    for alias in (
        "ZERODHA_API_KEY",
        "KITE_API_KEY",
        "ZERODHA_API_SECRET",
        "KITE_API_SECRET",
    ):
        monkeypatch.delenv(alias, raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "KiteConnect", _RecoveredKite)
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["zerodha_token_cli.py", "--recover-saved-token", "--env", str(env_file)],
    )
    return cli, env_file


def test_recover_saved_token_verifies_broker_and_queues_safe_restart(
    monkeypatch, tmp_path, capsys
):
    from nifty_scalper_bot import admin_dashboard
    from nifty_scalper_bot.ops.service_control import ControlResult

    cli, _env_file = _configure_saved_session(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        admin_dashboard,
        "_activate_access_token",
        lambda token: calls.append(token)
        or ControlResult(True, "restart_bot", "Validated restart queued"),
    )
    cli.main()
    assert calls == ["saved-token"]
    output = capsys.readouterr().out
    assert "broker funds verified" in output
    assert "saved-token" not in output
    assert "Validated restart queued" in output


def test_recover_saved_token_rejects_wrong_owner_without_rotating(
    monkeypatch, tmp_path
):
    from nifty_scalper_bot import admin_dashboard

    cli, _env_file = _configure_saved_session(monkeypatch, tmp_path)

    class WrongOwnerKite(_RecoveredKite):
        def profile(self):
            return {"user_id": "OTHER"}

    monkeypatch.setattr(cli, "KiteConnect", WrongOwnerKite)
    monkeypatch.setattr(
        admin_dashboard,
        "_activate_access_token",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not rotate")),
    )
    with pytest.raises(SystemExit, match="Saved session not verified"):
        cli.main()


def test_recover_saved_token_requires_broker_margin_response(monkeypatch, tmp_path):
    cli, _env_file = _configure_saved_session(monkeypatch, tmp_path)

    class NoMarginsKite(_RecoveredKite):
        def margins(self, segment):
            return {}

    monkeypatch.setattr(cli, "KiteConnect", NoMarginsKite)
    with pytest.raises(SystemExit, match="Saved session not verified"):
        cli.main()
