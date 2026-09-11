from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.broker_pnl_authority_patch import (
    _extract_account_m2m,
    _strategy_day_marked_pnl,
    _strip_legacy_position_pnl,
)
from nifty_scalper_bot.execution.position_manager import PositionManager


SYMBOL = "NFO:NIFTY2691523400CE"


def test_extract_account_m2m_accepts_zerodha_spelling_variants() -> None:
    realized, unrealized = _extract_account_m2m(
        {
            "utilised": {
                "m2m_realised": -125.5,
                "m2m_unrealised": 42.25,
            }
        }
    )
    assert realized == pytest.approx(-125.5)
    assert unrealized == pytest.approx(42.25)

    realized_us, unrealized_us = _extract_account_m2m(
        {
            "utilized": {
                "m2m_realized": 10.0,
                "m2m_unrealized": -2.0,
            }
        }
    )
    assert realized_us == pytest.approx(10.0)
    assert unrealized_us == pytest.approx(-2.0)


def test_strategy_day_pnl_uses_buy_sell_economics_not_legacy_realised() -> None:
    marked, closed, rows = _strategy_day_marked_pnl(
        [
            {
                "exchange": "NFO",
                "tradingsymbol": "NIFTY2691523400CE",
                "product": "MIS",
                "buy_value": 1000.0,
                "sell_value": 1200.0,
                "quantity": 10,
                "last_price": 50.0,
                "multiplier": 1,
                "realised": -999999.0,
            },
            {
                "exchange": "NFO",
                "tradingsymbol": "NIFTY2691523400PE",
                "product": "MIS",
                "buy_value": 500.0,
                "sell_value": 450.0,
                "quantity": 0,
                "last_price": 0.0,
                "multiplier": 1,
                "realised": 999999.0,
            },
            {
                "exchange": "NFO",
                "tradingsymbol": "BANKNIFTY2691540000CE",
                "product": "MIS",
                "buy_value": 1.0,
                "sell_value": 9999.0,
                "quantity": 0,
                "last_price": 0.0,
            },
        ]
    )

    # 1200 - 1000 + (10 * 50) = 700; closed PE = -50.
    assert marked == pytest.approx(650.0)
    assert closed == pytest.approx(-50.0)
    assert rows == 2


def test_strip_legacy_position_pnl_preserves_exposure_shape() -> None:
    payload = {
        "net": [
            {
                "symbol": SYMBOL,
                "quantity": 65,
                "product": "MIS",
                "realised": -999.0,
                "average_price": 100.0,
            }
        ],
        "day": [{"tradingsymbol": "keep-me"}],
    }

    cleaned = _strip_legacy_position_pnl(payload)

    assert cleaned["net"][0]["quantity"] == 65
    assert cleaned["net"][0]["average_price"] == 100.0
    assert "realised" not in cleaned["net"][0]
    assert cleaned["day"] == payload["day"]


def test_dedicated_broker_pnl_never_overwrites_strategy_ledger(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline()
    with manager._lock:
        manager._local_realized_pnl = -125.0
        manager._refresh_realized_pnl_locked()

    broker = SimpleNamespace(
        get_pnl_snapshot=lambda: {
            "account_realized": -90.0,
            "account_unrealized": -10.0,
            "account_total": -100.0,
            "strategy_day_marked_gross": -120.0,
            "strategy_day_closed_gross": -120.0,
            "strategy_day_rows": 1,
            "source": "zerodha_margins_m2m",
            "observed_at": "2026-09-11T10:00:00+00:00",
        }
    )
    manager.set_broker_client(broker)

    manager.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "product": "MIS",
                "quantity": 0,
                "average_price": 100.0,
                "last_price": 100.0,
                # This legacy field must not become strategy/account authority.
                "realised": -999.0,
            }
        ]
    )

    snapshot = manager.pnl_reconciliation_snapshot()
    assert manager.get_realized_pnl() == pytest.approx(-125.0)
    assert manager.get_strategy_realized_pnl() == pytest.approx(-125.0)
    assert manager.get_broker_account_realized_pnl() == pytest.approx(-90.0)
    assert snapshot["broker_account_realized"] == pytest.approx(-90.0)
    assert snapshot["broker_account_unrealized"] == pytest.approx(-10.0)
    assert snapshot["broker_vs_strategy_realized_difference"] == pytest.approx(35.0)
    assert snapshot["broker_account_pnl_status"] == "mismatch"
    assert snapshot["pnl_diagnostic_only"] is True
    assert snapshot["pnl_authority"] == "local_confirmed_ledger"
    assert snapshot["baseline_source"] == "zerodha_margins_m2m"
    assert snapshot["pnl_trading_date"] == manager._trading_date_ist()
    assert manager.current_pnl_reconciliation_blocker() is None


def test_broker_pnl_failure_preserves_local_strategy_accounting(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    with manager._lock:
        manager._local_realized_pnl = 77.0
        manager._refresh_realized_pnl_locked()

    def fail() -> dict[str, object]:
        raise RuntimeError("temporary margins outage")

    manager.set_broker_client(SimpleNamespace(get_pnl_snapshot=fail))
    manager.synchronize_with_broker([])

    snapshot = manager.pnl_reconciliation_snapshot()
    assert manager.get_realized_pnl() == pytest.approx(77.0)
    assert manager.get_broker_account_realized_pnl() is None
    assert snapshot["broker_account_pnl_error"] is not None
    assert snapshot["pnl_diagnostic_only"] is True
