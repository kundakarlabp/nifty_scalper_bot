"""Regression contract for the operator-only failed-LIVE-startup bootstrap."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/scripts/recover_failed_live_startup.sh"


def test_bootstrap_is_shell_valid_and_embedded_python_compiles() -> None:
    assert SCRIPT.exists()
    assert (
        subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, check=False
        ).returncode
        == 0
    )
    source = SCRIPT.read_text(encoding="utf-8")
    for name in ("RECOVER", "READY"):
        start = f"<<'PY_{name}'\n"
        end = f"\nPY_{name}"
        assert start in source
        body = source.split(start, 1)[1].split(end, 1)[0]
        compile(body, f"recovery_{name}", "exec")


def test_recovery_is_pinned_to_main_ci_and_uses_candidate_release_guard() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "git fetch --quiet origin main" in source
    assert "refs/remotes/origin/main" in source
    assert "nifty-ci-passed/" in source
    assert 'git show "$target:deploy/lightsail_release.sh"' in source
    assert 'bash "$release_file" --auto' in source
    assert "sudo systemctl restart" not in source
    assert "git reset --hard" not in source
    assert "RISK__CAPITAL=" not in source
    assert "ENABLE_LIVE=true" not in source
    assert "EXECUTION_MODE=SHADOW" not in source


def test_broker_fail_closed_proofs_precede_restart() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    required = (
        "LIVE startup requires validated broker balance",
        'live.get("bot_loaded") is False',
        'health.get("live_orders_armed") is False',
        "kite.profile()",
        "kite.margins(segment)",
        "kite.positions()",
        "kite.orders()",
        "for _ in range(2):",
        'float(row["quantity"]) == 0',
        '"token_sha256"',
        '"verified_at"',
        "auto_release_restart_pending",
        "auto_release_verified_token_rotation",
    )
    for marker in required:
        assert marker in source
    assert source.index("kite.profile()") < source.index("os.replace(temporary, proof)")
    assert source.index("kite.orders()") < source.index("os.replace(temporary, proof)")
    assert source.index("os.replace(temporary, proof)") < source.index(
        'bash "$release_file" --auto'
    )
