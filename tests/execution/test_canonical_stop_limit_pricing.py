"""Regression contracts for the canonical NSE option SL-limit price policy."""

from __future__ import annotations

import math

import pytest

from nifty_scalper_bot.utils.pricing import (
    canonical_stop_limit_prices,
    configured_stop_limit_buffer_pct,
)


@pytest.mark.parametrize("side, expected", [
    ("SELL", 95.0),
    ("BUY", 105.0),
])
def test_default_stop_limit_has_single_five_percent_buffer(
    monkeypatch, side: str, expected: float
) -> None:
    monkeypatch.delenv("STOP_LIMIT_BUFFER_PCT", raising=False)
    monkeypatch.delenv("EXCHANGE_PROTECTIVE_STOP_LIMIT_BUFFER_PCT", raising=False)
    trigger, limit = canonical_stop_limit_prices(100.0, side)
    assert trigger == 100.0
    assert limit == expected


def test_new_config_overrides_legacy_buffer_for_every_stop_path(monkeypatch) -> None:
    monkeypatch.setenv("EXCHANGE_PROTECTIVE_STOP_LIMIT_BUFFER_PCT", "7")
    monkeypatch.setenv("STOP_LIMIT_BUFFER_PCT", "3")
    assert configured_stop_limit_buffer_pct() == 3.0
    assert canonical_stop_limit_prices(100.0, "SELL") == (100.0, 97.0)
    assert canonical_stop_limit_prices(100.0, "BUY") == (100.0, 103.0)


def test_legacy_buffer_configuration_remains_compatible(monkeypatch) -> None:
    monkeypatch.delenv("STOP_LIMIT_BUFFER_PCT", raising=False)
    monkeypatch.setenv("EXCHANGE_PROTECTIVE_STOP_LIMIT_BUFFER_PCT", "6")
    assert canonical_stop_limit_prices(100.0, "SELL") == (100.0, 94.0)


@pytest.mark.parametrize("side", ["SELL", "BUY"])
def test_stop_trigger_and_limit_are_valid_nse_ticks(side: str) -> None:
    trigger, limit = canonical_stop_limit_prices(100.03, side, buffer_pct=5.0)
    assert trigger == 100.05
    assert math.isclose(round(trigger / 0.05) * 0.05, trigger)
    assert math.isclose(round(limit / 0.05) * 0.05, limit)
    if side == "SELL":
        assert 0 < limit < trigger
    else:
        assert limit > trigger


def test_lowest_positive_sell_trigger_cannot_form_valid_stop_limit() -> None:
    trigger, limit = canonical_stop_limit_prices(0.05, "SELL")
    assert trigger == limit == 0.05  # caller must reject this invalid geometry


@pytest.mark.parametrize("price, side", [(0.0, "SELL"), (math.nan, "BUY"), (100, "OTHER")])
def test_invalid_order_inputs_are_rejected(price: float, side: str) -> None:
    with pytest.raises(ValueError):
        canonical_stop_limit_prices(price, side)


def test_nonfinite_config_falls_back_to_safe_default(monkeypatch) -> None:
    monkeypatch.setenv("STOP_LIMIT_BUFFER_PCT", "nan")
    assert configured_stop_limit_buffer_pct() == 5.0
