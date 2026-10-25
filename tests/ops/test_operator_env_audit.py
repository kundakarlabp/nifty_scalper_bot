"""Test the read-only operator env audit never prints sensitive values."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "scripts" / "audit_operator_env.py"


def _run(path: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    response = subprocess.run(
        [sys.executable, str(SCRIPT), "--env-file", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return response, json.loads(response.stdout)


def _private_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def test_reports_only_known_policy_keys_and_no_values(tmp_path: Path) -> None:
    path = tmp_path / "niftybot.env"
    secret = "ULTRA_PRIVATE_BROKER_ACCESS_TOKEN_123"
    _private_file(
        path,
        "\n".join(
            [
                f"KITE_ACCESS_TOKEN={secret}",
                "ENABLE_LIVE=true",
                "EXECUTION_MODE=LIVE",
                "GIT_COMMIT_SHA=" + "a" * 40,
                "RISK_PER_TRADE_PCT=2.0",
                "SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED=false",
                "SUPABASE_RUNTIME_HEARTBEAT_URL=https://private-endpoint.example",
                "CUSTOM_PASSPHRASE=OTHER_PRIVATE_VALUE",
                "CUSTOM_METADATA=NONPUBLIC_OTHER_VALUE",
            ]
        ),
    )
    before = path.read_bytes()

    response, payload = _run(path)

    assert response.returncode == 0
    assert payload["status"] == "ok"
    assert payload["mode_alignment"] == "explicit_live"
    assert payload["strategy_risk_overrides"] == [
        "RISK_PER_TRADE_PCT",
        "SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED",
    ]
    assert payload["monitoring_overrides"] == ["SUPABASE_RUNTIME_HEARTBEAT_URL"]
    assert payload["duplicate_policy_keys"] == []
    assert payload["unclassified_key_count"] == 1
    assert secret not in response.stdout + response.stderr
    assert "OTHER_PRIVATE_VALUE" not in response.stdout + response.stderr
    assert "NONPUBLIC_OTHER_VALUE" not in response.stdout + response.stderr
    assert "2.0" not in response.stdout + response.stderr
    assert "https://private-endpoint" not in response.stdout + response.stderr
    assert str(path) not in response.stdout + response.stderr
    assert path.read_bytes() == before


def test_conflicting_mode_and_duplicate_policy_are_diagnostic_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "secrets.env"
    _private_file(
        path,
        "ENABLE_LIVE=true\nEXECUTION_MODE=SHADOW\n"
        "RISK_DAILY_LOSS_PCT=3\nRISK_DAILY_LOSS_PCT=2\n",
    )

    response, payload = _run(path)

    assert response.returncode == 0
    assert payload["status"] == "ok"
    assert payload["mode_alignment"] == "conflict"
    assert payload["duplicate_policy_keys"] == ["RISK_DAILY_LOSS_PCT"]
    assert payload["strategy_risk_overrides"] == ["RISK_DAILY_LOSS_PCT"]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("ENABLE_LIVE=false\nEXECUTION_MODE=SHADOW\n", "explicit_nonlive"),
        ("ENABLE_LIVE=false\nEXECUTION_MODE=PAPER\n", "explicit_nonlive"),
        ("EXECUTION_MODE=LIVE\n", "incomplete"),
        ("ENABLE_LIVE=true\n", "incomplete"),
        ("ENABLE_LIVE=true\nEXECUTION_MODE=LIVE\nENABLE_LIVE=false\n", "ambiguous"),
    ],
)
def test_mode_classification(tmp_path: Path, content: str, expected: str) -> None:
    path = tmp_path / "niftybot.env"
    _private_file(path, content)
    response, payload = _run(path)
    assert response.returncode == 0
    assert payload["mode_alignment"] == expected


def test_export_prefixed_settings_remain_visible_as_names(tmp_path: Path) -> None:
    path = tmp_path / "niftybot.env"
    _private_file(path, "export RISK_PER_TRADE_PCT=3.0\\n")
    response, payload = _run(path)
    assert response.returncode == 0
    assert payload["strategy_risk_overrides"] == ["RISK_PER_TRADE_PCT"]
    assert payload["malformed_line_count"] == 0


def test_whitespace_around_equal_matches_dotenv(tmp_path: Path) -> None:
    path = tmp_path / "niftybot.env"
    _private_file(
        path,
        "ENABLE_LIVE = true\\nEXECUTION_MODE = LIVE\\nRISK_PER_TRADE_PCT = 0.5\\n",
    )
    response, payload = _run(path)
    assert response.returncode == 0
    assert payload["mode_alignment"] == "explicit_live"
    assert payload["strategy_risk_overrides"] == ["RISK_PER_TRADE_PCT"]
    assert payload["malformed_line_count"] == 0
    assert "0.5" not in response.stdout + response.stderr


def test_fifo_returns_unsafe_result_without_hanging(tmp_path: Path) -> None:
    fifo = tmp_path / "stream.env"
    os.mkfifo(fifo, mode=0o600)
    response, payload = _run(fifo)
    assert response.returncode != 0
    assert payload == {"status": "missing_or_unsafe_file"}


def test_missing_file_fails_closed_without_leaking_path(tmp_path: Path) -> None:
    path = tmp_path / "secret-user-name.env"
    response, payload = _run(path)
    assert response.returncode != 0
    assert payload == {"status": "missing_or_unsafe_file"}
    assert str(path) not in response.stdout + response.stderr


def test_symlink_and_world_readable_file_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "private.env"
    _private_file(target, "ENABLE_LIVE=true\nEXECUTION_MODE=LIVE\n")
    link = tmp_path / "redirected.env"
    link.symlink_to(target)
    response, payload = _run(link)
    assert response.returncode != 0
    assert payload == {"status": "missing_or_unsafe_file"}

    target.chmod(0o644)
    response, payload = _run(target)
    assert response.returncode != 0
    assert payload == {"status": "missing_or_unsafe_file"}


def test_documented_shell_command_uses_one_continuation() -> None:
    runbook = (ROOT / "docs" / "ops" / "nifty_single_path_runbook.md").read_text(
        encoding="utf-8"
    )
    line = next(
        x
        for x in runbook.splitlines()
        if ".venv/bin/python deploy/scripts/audit_operator_env.py" in x
    )
    assert line.endswith(chr(92))
    assert not line.endswith(chr(92) * 2)


def test_malformed_or_oversize_payload_never_echoes_sensitive_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "niftybot.env"
    _private_file(path, "INVALID=" + ("DO_NOT_LEAK" * 25000))
    response, payload = _run(path)
    assert response.returncode != 0
    assert payload == {"status": "missing_or_unsafe_file"}
    assert "DO_NOT_LEAK" not in response.stdout + response.stderr
