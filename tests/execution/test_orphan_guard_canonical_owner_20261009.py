"""Regression tests for 9 Oct managed-position/orphan-guard false positives."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from nifty_scalper_bot.execution.order_manager_core import OrderManager


SYMBOL = "NFO:NIFTY26O1322450CE"


def _guard_fixture() -> tuple[SimpleNamespace, MagicMock]:
    bracket = MagicMock()
    bracket.is_symbol_managed.return_value = False
    bracket.get_bracket.return_value = None
    broker = MagicMock()
    broker.get_positions.return_value = [
        {"symbol": SYMBOL, "quantity": 65, "average_price": 144.15}
    ]
    manager = SimpleNamespace(
        _bracket_manager=bracket,
        _market_data=None,
        _data_hub=None,
        _broker=broker,
        _logger=MagicMock(),
        _adopt_orphan_position=MagicMock(),
        _log_trade_event=MagicMock(),
    )
    return manager, bracket


def test_managed_symbol_never_creates_a_synthetic_guard() -> None:
    manager, bracket = _guard_fixture()
    bracket.is_symbol_managed.return_value = True

    assert OrderManager.guard_orphan_position(manager, SYMBOL, 65, 144.15) is True
    bracket.register_virtual_bracket.assert_not_called()
    bracket.confirm_entry_fill.assert_not_called()
    manager._adopt_orphan_position.assert_not_called()


def test_rejected_orphan_registration_does_not_claim_attachment() -> None:
    manager, bracket = _guard_fixture()

    assert OrderManager.guard_orphan_position(manager, SYMBOL, 65, 144.15) is False
    bracket.register_virtual_bracket.assert_called_once()
    bracket.confirm_entry_fill.assert_not_called()
    manager._log_trade_event.assert_not_called()
    assert not any(
        "ORPHAN_POSITION_BRACKET_ATTACHED" in str(call)
        for call in manager._logger.info.call_args_list
    )


def test_concurrent_canonical_owner_wins_without_phantom_guard() -> None:
    manager, bracket = _guard_fixture()
    # Prechecks say orphan; the competing fill registers its real owner
    # before the synthetic guard can be installed.
    bracket.is_symbol_managed.side_effect = [False, False, True]

    assert OrderManager.guard_orphan_position(manager, SYMBOL, 65, 144.15) is True
    bracket.register_virtual_bracket.assert_called_once()
    bracket.confirm_entry_fill.assert_not_called()
    manager._log_trade_event.assert_not_called()


def test_guard_reports_attached_only_after_confirmed_activation() -> None:
    manager, bracket = _guard_fixture()
    bracket.get_bracket.return_value = SimpleNamespace(entry_confirmed=True)

    assert OrderManager.guard_orphan_position(manager, SYMBOL, 65, 144.15) is True
    bracket.confirm_entry_fill.assert_called_once()
    manager._log_trade_event.assert_called_once()
    assert any(
        "ORPHAN_POSITION_BRACKET_ATTACHED" in str(call)
        for call in manager._logger.info.call_args_list
    )
