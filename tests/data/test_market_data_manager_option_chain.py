from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class _StubResolver:
    def __init__(self, contracts: list[dict[str, Any]]) -> None:
        self._contracts = contracts

    def option_contracts(
        self, underlying: str, *, force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        return [dict(row) for row in self._contracts]


class _StubBroker:
    def __init__(
        self, quotes: dict[int, dict[str, Any]], underlying_price: float | None = None
    ) -> None:
        self._quotes = {int(token): dict(payload) for token, payload in quotes.items()}
        self.bulk_calls: list[tuple[int, ...]] = []
        self.quote_calls: list[str] = []
        self._underlying_price = underlying_price

    def get_quote_bulk(self, tokens: list[int]) -> dict[int, dict[str, Any]]:
        self.bulk_calls.append(tuple(int(token) for token in tokens))
        return {
            token: dict(self._quotes[token])
            for token in tokens
            if token in self._quotes
        }

    def get_quote(self, symbol: str) -> dict[str, Any]:
        self.quote_calls.append(symbol)
        if self._underlying_price is None:
            raise RuntimeError("price unavailable")
        return {"ltp": float(self._underlying_price)}


@pytest.fixture()
def sample_contracts() -> list[dict[str, Any]]:
    expiry = (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat()
    return [
        {
            "instrument_token": 501,
            "tradingsymbol": "NIFTY25JAN20000CE",
            "option_type": "CE",
            "expiry": expiry,
            "strike": 20000,
        },
        {
            "instrument_token": 502,
            "tradingsymbol": "NIFTY25JAN20000PE",
            "option_type": "PE",
            "expiry": expiry,
            "strike": 20000,
        },
        {
            "instrument_token": 503,
            "tradingsymbol": "NIFTY25JAN20100CE",
            "option_type": "CE",
            "expiry": expiry,
            "strike": 20100,
        },
        {
            "instrument_token": 504,
            "tradingsymbol": "NIFTY25JAN20100PE",
            "option_type": "PE",
            "expiry": expiry,
            "strike": 20100,
        },
    ]


def test_option_chain_ranks_and_enriches(
    sample_contracts: list[dict[str, Any]],
) -> None:
    quotes = {
        501: {
            "last_price": 120.5,
            "depth": {
                "buy": [{"price": 120.0, "quantity": 10}],
                "sell": [{"price": 121.0, "quantity": 12}],
            },
            "open_interest": 1500,
            "volume_traded_today": 450,
        },
        502: {
            "last_price": 118.0,
            "depth": {
                "buy": [{"price": 117.5, "quantity": 8}],
                "sell": [{"price": 118.5, "quantity": 9}],
            },
            "oi": 1600,
            "volume_traded": 430,
        },
        503: {"last_price": 90.0},
        504: {"last_price": 92.0},
    }
    broker = _StubBroker(quotes)
    resolver = _StubResolver(sample_contracts)
    manager = MarketDataManager(broker, None, resolver=resolver)
    manager._latest_ticks["NIFTY"] = {"ltp": 20010.0}  # type: ignore[attr-defined]

    chain = manager.get_option_chain("weekly")
    assert chain is not None
    assert {row["option_type"] for row in chain} == {"CE", "PE"}
    assert chain[0]["strike"] <= chain[-1]["strike"]
    assert chain[0]["ltp"] > 0
    assert broker.bulk_calls  # ensures quotes fetched in batch


def test_option_chain_monthly_expiry_uses_broker_quote() -> None:
    base = datetime.now(timezone.utc)
    contracts = [
        {
            "instrument_token": 600,
            "tradingsymbol": "NIFTY25JAN20000CE",
            "option_type": "CE",
            "expiry": (base + timedelta(days=3)).date().isoformat(),
            "strike": 20000,
        },
        {
            "instrument_token": 601,
            "tradingsymbol": "NIFTY25JAN20000PE",
            "option_type": "PE",
            "expiry": (base + timedelta(days=24)).date().isoformat(),
            "strike": 20000,
        },
    ]
    quotes = {
        600: {"last_price": 110.0},
        601: {"last_price": 95.0},
    }
    broker = _StubBroker(quotes, underlying_price=19975.0)
    resolver = _StubResolver(contracts)
    manager = MarketDataManager(broker, None, resolver=resolver)

    chain = manager.get_option_chain("monthly")
    assert chain is not None
    assert all(row["expiry"].date() == chain[0]["expiry"].date() for row in chain)
    assert broker.quote_calls == ["NIFTY"]


def test_option_chain_handles_expiry_with_time_component() -> None:
    base = datetime.now(timezone.utc) + timedelta(days=3)
    formatted_expiry = base.strftime("%d-%b-%Y %H:%M:%S")
    contracts = [
        {
            "instrument_token": 701,
            "tradingsymbol": "NIFTY25JAN20000CE",
            "option_type": "CE",
            "expiry": formatted_expiry,
            "strike": 20000,
        },
        {
            "instrument_token": 702,
            "tradingsymbol": "NIFTY25JAN20000PE",
            "option_type": "PE",
            "expiry": formatted_expiry,
            "strike": 20000,
        },
    ]
    quotes = {
        701: {"last_price": 105.0},
        702: {"last_price": 98.0},
    }
    broker = _StubBroker(quotes)
    resolver = _StubResolver(contracts)
    manager = MarketDataManager(broker, None, resolver=resolver)
    manager._latest_ticks["NIFTY"] = {"ltp": 20010.0}  # type: ignore[attr-defined]

    chain = manager.get_option_chain("weekly")
    assert chain is not None
    assert {row["instrument_token"] for row in chain} == {701, 702}
    assert all(row["expiry"].date() == base.date() for row in chain)


def test_option_chain_falls_back_to_raw_zerodha_nfo_instruments() -> None:
    expiry_dt = (datetime.now(timezone.utc) + timedelta(days=3)).date()

    class EmptyResolver:
        def option_contracts(self, underlying: str, *, force_refresh: bool = False) -> list[dict[str, Any]]:
            return []

    class RawBroker(_StubBroker):
        def instruments(self, exchange: str) -> list[dict[str, Any]]:
            assert exchange == "NFO"
            return [
                {
                    "name": "NIFTY",
                    "instrument_type": "CE",
                    "tradingsymbol": "NIFTY26JUN25000CE",
                    "instrument_token": 901,
                    "expiry": datetime.combine(expiry_dt, datetime.min.time()),
                    "strike": 25000,
                    "lot_size": 25,
                    "tick_size": 0.05,
                },
                {
                    "name": "NIFTY",
                    "instrument_type": "PE",
                    "tradingsymbol": "NIFTY26JUN25000PE",
                    "instrument_token": 902,
                    "expiry": expiry_dt.isoformat(),
                    "strike": 25000,
                    "lot_size": 25,
                    "tick_size": 0.05,
                },
                {
                    "name": "NIFTY",
                    "instrument_type": "FUT",
                    "tradingsymbol": "NIFTY26JUNFUT",
                    "instrument_token": 903,
                    "expiry": expiry_dt.isoformat(),
                    "strike": 0,
                },
            ]

    broker = RawBroker(
        {
            901: {"last_price": 110.0, "depth": {"buy": [{"price": 109.5}], "sell": [{"price": 110.5}]}},
            902: {"last_price": 111.0, "depth": {"buy": [{"price": 110.5}], "sell": [{"price": 111.5}]}},
        },
        underlying_price=25010.0,
    )
    manager = MarketDataManager(broker, None, resolver=EmptyResolver())

    chain = manager.get_option_chain(expiry_dt.isoformat())

    assert chain is not None
    assert {row["instrument_token"] for row in chain} == {901, 902}
    assert {row["option_type"] for row in chain} == {"CE", "PE"}
    assert all(row["expiry"].date() == expiry_dt for row in chain)
