from __future__ import annotations

import math
from datetime import datetime

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.utils.options_math import black_scholes_price


class DummyMDM:
    def pull_quote(
        self, symbol: str
    ) -> dict[str, float] | None:  # pragma: no cover - not used
        return None


class DummyResolver:
    def lot_size_for_symbol(
        self, symbol: str
    ) -> int:  # pragma: no cover - compatibility
        return 75


def _make_hub(now: datetime) -> DataHub:
    ts = now.timestamp()
    return DataHub(
        DummyMDM(),
        DummyResolver(),
        clock=lambda: ts,
    )


def test_iv_fallback_and_expected_move() -> None:
    now = datetime(2024, 9, 2, 9, 20)
    hub = _make_hub(now)
    spot_price = 20000.0
    hub.ingest_tick({"symbol": "NIFTY", "ltp": spot_price})
    symbol = "NIFTY24SEP20000CE"
    expiry = datetime(2024, 9, 28, 15, 30)
    ttm_years = (expiry - now).total_seconds() / 31557600.0
    iv_true = 0.2
    option_price = black_scholes_price(
        spot_price,
        20000.0,
        ttm_years,
        0.06,
        iv_true,
        is_call=True,
    )
    hub.ingest_tick(
        {
            "symbol": symbol,
            "ltp": option_price,
            "open_interest": 12345,
        }
    )

    iv = hub.get_iv(symbol)
    assert iv is not None
    assert math.isclose(iv, iv_true, rel_tol=1e-3)

    greeks = hub.get_greeks(symbol)
    assert greeks is not None
    assert 0.0 < greeks["delta"] < 1.0

    oi = hub.get_oi(symbol)
    assert oi == 12345

    expected = spot_price * iv_true * math.sqrt(1.0 / 365.0)
    move = hub.expected_move(1.0)
    assert move is not None
    assert math.isclose(move, expected, rel_tol=1e-3)


def test_iv_direct_value() -> None:
    now = datetime(2024, 9, 2, 9, 20)
    hub = _make_hub(now)
    hub.ingest_tick({"symbol": "NIFTY", "ltp": 20123.0})
    symbol = "NIFTY24SEP20100PE"
    hub.ingest_tick(
        {
            "symbol": symbol,
            "ltp": 105.0,
            "implied_volatility": 0.35,
            "greeks": {"delta": -0.42, "theta": -50.0, "vega": 120.0},
        }
    )

    iv = hub.get_iv(symbol)
    assert iv is not None
    assert math.isclose(iv, 0.35, rel_tol=1e-6)
    greeks = hub.get_greeks(symbol)
    assert greeks is not None
    assert greeks["delta"] < 0
