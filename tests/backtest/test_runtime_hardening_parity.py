"""Regression contract: replay must install the same runtime hardening as LIVE."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import nifty_scalper_bot.core as core
from nifty_scalper_bot.backtest.replay import ReplayHarness


_REQUIRED = {
    "market_data_hardening",
    "dynamic_universe",
    "live_ws_tick_receipts",
    "runtime_reliability",
    "runner_candle_cache",
    "strategy_context_fast_path",
    "off_market_controller",
    "off_market_app",
    "session_boundary",
    "boot_readiness",
    "polling_failover",
}


def test_core_exports_explicit_runtime_hardening_installer() -> None:
    installer = getattr(core, "install_runtime_hardening", None)
    assert callable(installer)


def test_runtime_hardening_install_proof_is_complete() -> None:
    installer = getattr(core, "install_runtime_hardening", None)
    assert callable(installer)

    proof = installer()

    assert _REQUIRED <= set(proof)
    assert all(proof[name] is True for name in _REQUIRED)


def test_runtime_hardening_cold_import_does_not_recurse() -> None:
    """A pristine interpreter must install hardening without package self-recursion."""
    script = "\n".join(
        (
            "import json",
            "from nifty_scalper_bot.core import install_runtime_hardening",
            "proof = install_runtime_hardening()",
            "print('RUNTIME_PROOF=' + json.dumps(proof, sort_keys=True))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"

    assert completed.returncode == 0, output
    assert "RecursionError" not in output, output
    assert "Failure in core.__getattr__" not in output, output

    proof_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith("RUNTIME_PROOF=")),
        None,
    )
    assert proof_line is not None, output
    proof = json.loads(proof_line.removeprefix("RUNTIME_PROOF="))
    assert _REQUIRED <= set(proof)
    assert all(proof[name] is True for name in _REQUIRED)


def test_replay_harness_calls_explicit_runtime_hardening(monkeypatch) -> None:
    calls: list[str] = []

    def _install() -> dict[str, bool]:
        calls.append("installed")
        return {name: True for name in _REQUIRED}

    monkeypatch.setattr(core, "install_runtime_hardening", _install, raising=False)

    ReplayHarness(
        runner=SimpleNamespace(),
        paper_engine=SimpleNamespace(),
        option_symbol="NFO:NIFTY26SEP24000CE",
    )

    assert calls == ["installed"]


def test_replay_harness_fails_closed_when_runtime_hardening_is_incomplete(
    monkeypatch,
) -> None:
    def _fail() -> dict[str, bool]:
        raise RuntimeError("runtime_hardening_incomplete missing=['polling_failover']")

    monkeypatch.setattr(core, "install_runtime_hardening", _fail, raising=False)

    with pytest.raises(RuntimeError, match="runtime_hardening_incomplete"):
        ReplayHarness(
            runner=SimpleNamespace(),
            paper_engine=SimpleNamespace(),
            option_symbol="NFO:NIFTY26SEP24000CE",
        )
