"""Tests for Telegram command analytics formatting helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.notifications.telegram_commands import (
    _format_chain_summary,
    _format_orderbook_snapshot,
    _parse_chain_argument,
)
from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    StrategyRunnerConfig,
    TradeRecord,
)


def test_format_chain_summary_includes_metrics() -> None:
    chain = [
        {
            "strike": 20000,
            "option_type": "CE",
            "open_interest": 1200,
            "trades": 450,
            "bid": 12.5,
            "ask": 13.5,
            "iv": 14.2,
            "depth": {"buy": [{"price": 12.5, "quantity": 75}], "sell": []},
            "expiry": "2024-06-06",
        },
        {
            "strike": 20000,
            "option_type": "PE",
            "open_interest": 800,
            "trades": 300,
            "bid": 9.0,
            "ask": 10.0,
            "iv": 15.7,
            "depth": {"buy": [], "sell": [{"price": 10.0, "quantity": 60}]},
            "expiry": "2024-06-06",
        },
        {
            "strike": 20100,
            "option_type": "CE",
            "open_interest": 500,
            "trades": 250,
            "bid": 10.0,
            "ask": 11.5,
            "iv": 13.1,
            "depth": {"buy": [{"price": 10.0, "quantity": 40}], "sell": []},
            "expiry": "2024-06-06",
        },
        {
            "strike": 20100,
            "option_type": "PE",
            "open_interest": 700,
            "trades": 275,
            "bid": 8.5,
            "ask": 9.5,
            "iv": 16.2,
            "depth": {"buy": [], "sell": [{"price": 9.5, "quantity": 55}]},
            "expiry": "2024-06-06",
        },
    ]

    summary = _format_chain_summary(chain, "NIFTY", "2024-06-06")

    assert "total_oi=" in summary
    assert "Volume heatmap:" in summary
    assert "Liquidity focus:" in summary
    assert "IV% CE=" in summary


def test_format_orderbook_snapshot_includes_liquidity() -> None:
    snapshot = {
        "levels": 3,
        "best_bid": 199.5,
        "best_ask": 200.5,
        "spread": 1.0,
        "mid": 200.0,
        "total_bid_qty": 150.0,
        "total_ask_qty": 120.0,
        "liquidity_score": 270.0,
        "buy": [
            {"price": 199.5, "quantity": 75},
            {"price": 199.0, "quantity": 50},
        ],
        "sell": [
            {"price": 200.5, "quantity": 60},
            {"price": 201.0, "quantity": 45},
        ],
    }

    formatted = _format_orderbook_snapshot("NIFTY24MAYFUT", snapshot)

    assert "order book" in formatted
    assert "bid=199.50" in formatted
    assert "sell: 1:200.50@60" in formatted


def test_market_data_get_orderbook_normalizes_levels() -> None:
    broker = MagicMock()
    manager = MarketDataManager(broker)
    quote_payload = {
        "depth": {
            "buy": [
                {"price": 100, "quantity": 25},
                {"price": 99.5, "quantity": 15},
            ],
            "sell": [
                {"price": 100.5, "quantity": 20},
                {"price": 101, "quantity": 10},
            ],
        }
    }
    manager.get_quote = MagicMock(return_value=quote_payload)

    snapshot = manager.get_orderbook("nifty24mayfut", levels=2)

    assert snapshot is not None
    assert snapshot["best_bid"] == 100.0
    assert snapshot["best_ask"] == 100.5
    assert snapshot["spread"] == pytest.approx(0.5)
    assert snapshot["buy"][0]["quantity"] == 25.0
    assert snapshot["sell"][1]["price"] == 101.0


def test_log_skip_tags_canonicalizes_reason(caplog: pytest.LogCaptureFixture) -> None:
    manager = MagicMock()
    indicator = MagicMock()
    strategy_manager = MagicMock()
    risk_manager = MagicMock()
    order_manager = MagicMock()
    position_manager = MagicMock()

    config = StrategyRunnerConfig(max_trade_history=2)
    runner_instance = StrategyRunner(
        market_data_manager=manager,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        config=config,
        data_hub=None,
        strike_selector=None,
    )
    runner_instance.add_symbol("NIFTY")

    record = TradeRecord(
        timestamp=datetime.now(timezone.utc),
        action="BUY",
        quantity=1,
        price=100.0,
        status="skipped",
        reason="Cooldown active for 1s",
        order_id=None,
        reason_tags={"rule": "Cooldown active for 1s", "value": 1},
    )

    with caplog.at_level("INFO"):
        runner_instance._log_skip_tags("NIFTY", record)

    messages = [record.getMessage() for record in caplog.records]
    assert any("COOLDOWN" in message for message in messages)
    assert all("Cooldown active" not in message for message in messages)


def test_parse_chain_argument_handles_delimited() -> None:
    underlying, expiry = _parse_chain_argument("BANKNIFTY:2024-06-06")
    assert underlying == "BANKNIFTY"
    assert expiry == "2024-06-06"
