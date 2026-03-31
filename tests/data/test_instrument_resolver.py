"""Unit tests for :mod:`nifty_scalper_bot.data.instruments`."""

from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.data import InstrumentResolver


class _StubBroker:
    """Stub broker exposing instrument dumps for resolver tests."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def list_instruments(self) -> list[dict[str, Any]]:
        """Return static instrument dump for tests."""

        return list(self._rows)


@pytest.fixture()
def resolver() -> InstrumentResolver:
    """Return a resolver warmed with stubbed instruments."""

    rows = [
        {
            "tradingsymbol": "NIFTY25OCT25900CE",
            "exchange": "NFO",
            "instrument_token": 12345,
            "lot_size": 75,
        }
    ]
    broker = _StubBroker(rows)
    inst_resolver = InstrumentResolver(broker)
    inst_resolver.warm_from_broker_dump(rows)
    return inst_resolver


def test_resolve_symbol_to_token(resolver: InstrumentResolver) -> None:
    """Resolver should map option symbols to instrument tokens."""

    token = resolver.resolve_symbol_to_token("NIFTY25OCT25900CE")
    assert token == 12345


def test_build_quote_keys(resolver: InstrumentResolver) -> None:
    """Resolver should expose canonical quote keys for broker lookups."""

    symbol, candidates = resolver.build_quote_keys("Nifty25Oct25900ce")
    assert symbol == "NIFTY25OCT25900CE"
    assert candidates[0] == "NFO:NIFTY25OCT25900CE"
    assert "NIFTY25OCT25900CE" in candidates


def test_canonicalize_option(resolver: InstrumentResolver) -> None:
    """Canonicalization should return exchange and segment for options."""

    symbol, exchange, segment = resolver.canonicalize("nfo:nifty25oct25900ce")
    assert symbol == "NIFTY25OCT25900CE"
    assert exchange == "NFO"
    assert segment == "OPTIONS"


def test_canonicalize_index(resolver: InstrumentResolver) -> None:
    """Canonicalization should provide sensible defaults for indices."""

    symbol, exchange, segment = resolver.canonicalize("nifty")
    assert symbol == "NIFTY"
    assert exchange in {"NSE", None}
    assert segment in {"INDEX", "UNKNOWN"}
