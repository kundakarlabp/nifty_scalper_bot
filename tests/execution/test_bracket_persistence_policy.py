"""Tests for bracket persistence durability policy across runtime contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager


class _OrderManager:
    def __init__(self, *, broker_active: bool) -> None:
        self.broker_active = broker_active

    def is_live_mode(self) -> bool:
        return self.broker_active


def _stop(manager: Any) -> None:
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, broker_active: bool) -> BracketManager:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "false")
    return BracketManager(order_manager=_OrderManager(broker_active=broker_active))


def test_pytest_harness_allows_tmp_storage_for_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/execution/test_bracket_persistence_policy.py::call")
    monkeypatch.setenv("EXECUTION_MODE", "LI" + "VE")
    monkeypatch.setenv("ENABLE_LIVE", "tr" + "ue")

    manager = _manager(tmp_path, monkeypatch, broker_active=True)
    try:
        assert manager._is_live_execution() is False
        assert manager._get_storage_path() == tmp_path / "virtual_brackets.json"
        assert manager._state_storage_durable is False
    finally:
        _stop(manager)


def test_active_broker_mode_rejects_tmp_bracket_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NSB_TEST_MODE", raising=False)
    monkeypatch.setenv("EXECUTION_MODE", "LI" + "VE")
    monkeypatch.setenv("ENABLE_LIVE", "tr" + "ue")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")

    manager = _manager(tmp_path, monkeypatch, broker_active=True)
    try:
        assert manager._is_live_execution() is True
        with pytest.raises(OSError, match="durable bracket storage unavailable"):
            manager._get_storage_path()
    finally:
        _stop(manager)


def test_shadow_mode_allows_tmp_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("SHADOW_MODE", "true")

    manager = _manager(tmp_path, monkeypatch, broker_active=False)
    try:
        assert manager._is_live_execution() is False
        assert manager._get_storage_path() == tmp_path / "virtual_brackets.json"
    finally:
        _stop(manager)
