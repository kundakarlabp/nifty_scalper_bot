"""Tests for instrument resolver fallbacks."""

import pytest

from nifty_scalper_bot.data.instruments import InstrumentResolver
from nifty_scalper_bot.utils.errors import BrokerError


class _NoopBroker:
    """Broker stub that exposes no instrument listing APIs."""


class _InstrumentBroker:
    """Broker stub returning static instrument metadata."""

    def __init__(self, instruments: list[dict[str, object]]) -> None:
        self._instruments = instruments
        self.calls: list[str] = []

    def load_instruments(self, exchange: str) -> list[dict[str, object]]:
        self.calls.append(exchange)
        return list(self._instruments)


def test_format_token_as_symbol_uses_canonical_index_symbol() -> None:
    resolver = InstrumentResolver(_NoopBroker())
    resolver.warm()

    assert resolver.resolve("NIFTY") == 256265
    assert resolver.format_token_as_symbol(256265) == "NSE:NIFTY 50"


def test_format_token_as_symbol_handles_banknifty_alias() -> None:
    resolver = InstrumentResolver(_NoopBroker())
    resolver.warm()

    assert resolver.resolve("BANKNIFTY") == 260105
    assert resolver.format_token_as_symbol(260105) == "NSE:NIFTY BANK"


def test_tradingsymbol_for_order_requires_option_suffix() -> None:
    resolver = InstrumentResolver(_NoopBroker())

    with pytest.raises(BrokerError):
        resolver.tradingsymbol_for_order("NIFTY")


def test_exchange_for_symbol_enforces_nfo() -> None:
    resolver = InstrumentResolver(_NoopBroker())

    with pytest.raises(BrokerError):
        resolver.exchange_for_symbol("NSE:NIFTY25OCT20000CE")


def test_option_resolution_normalizes_symbol() -> None:
    resolver = InstrumentResolver(_NoopBroker())

    assert (
        resolver.tradingsymbol_for_order("nfo: nifty25oct20000ce")
        == "NIFTY25OCT20000CE"
    )
    assert resolver.exchange_for_symbol("nifty25oct20000pe") == "NFO"


def test_option_contracts_filters_and_caches() -> None:
    instruments = [
        {
            "tradingsymbol": "NIFTY25JAN20000CE",
            "instrument_token": 101,
            "instrument_type": "CE",
            "name": "NIFTY",
            "expiry": "2025-01-09",
            "strike": "20000",
            "lot_size": "50",
            "tick_size": "0.05",
        },
        {
            "tradingsymbol": "NIFTY25JAN20000PE",
            "instrument_token": 102,
            "instrument_type": "PE",
            "name": "NIFTY",
            "expiry": "2025-01-09",
            "strike": "20000",
        },
        {
            "tradingsymbol": "BANKNIFTY25JAN45000CE",
            "instrument_token": 201,
            "instrument_type": "CE",
            "name": "NIFTY BANK",
            "expiry": "2025-01-09",
            "strike": "45000",
        },
        {
            "tradingsymbol": "NIFTY25JANFUT",
            "instrument_token": 999,
            "instrument_type": "FUT",
            "name": "NIFTY",
            "expiry": "2025-01-09",
        },
    ]
    broker = _InstrumentBroker(instruments)
    resolver = InstrumentResolver(broker)

    contracts = resolver.option_contracts("NIFTY")
    tokens = {entry["instrument_token"] for entry in contracts}
    assert tokens == {101, 102}
    assert broker.calls == ["NFO"]

    # Cached call should not hit broker again
    resolver.option_contracts("NIFTY")
    assert broker.calls == ["NFO"]

    # Force refresh triggers a reload
    resolver.option_contracts("NIFTY", force_refresh=True)
    assert broker.calls == ["NFO", "NFO"]
