from typing import Any, Mapping

import pytest

from nifty_scalper_bot.config import settings as settings_module
from nifty_scalper_bot.core.unified_manager import UnifiedManager


class DummyResolver:
    def __init__(self) -> None:
        self._token_map: dict[str, int] = {}

    def tradingsymbol_for_order(self, symbol: str) -> str:
        return symbol.replace(" ", "").split(":", maxsplit=1)[-1]

    def exchange_for_symbol(self, symbol: str) -> str:
        return "NFO"

    def lot_size_for_symbol(self, symbol: str) -> int:
        return 25

    def resolve_symbol_to_token(self, symbol: str) -> int | None:
        return self._token_map.get(symbol.upper())

    def _ingest_instrument_row(self, row: Mapping[str, Any]) -> None:
        token = row.get("instrument_token")
        if token is None:
            return
        try:
            token_int = int(token)
        except (TypeError, ValueError):
            return
        tradingsymbol = str(row.get("tradingsymbol") or "").upper()
        if not tradingsymbol:
            return
        self._token_map.setdefault(tradingsymbol, token_int)
        self._token_map.setdefault(tradingsymbol.upper(), token_int)


class DummyMDM:
    def __init__(
        self, resolver: DummyResolver, tick: Mapping[str, Any] | None = None
    ) -> None:
        self._resolver = resolver
        self._latest_tick = tick

    def get_latest_tick(self, symbol: str) -> Mapping[str, Any] | None:
        return self._latest_tick

    def ingest_rest_quote(self, symbol: str, payload: Mapping[str, Any]) -> None:
        self._latest_tick = dict(payload)


class DummyBroker:
    def __init__(self, quote_payload: Mapping[str, Any]) -> None:
        self._payload = quote_payload

    def quote(self, _keys: list[str]) -> Mapping[str, Any]:
        return self._payload


@pytest.fixture(autouse=True)
def reset_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    settings_module.get_settings.cache_clear()
    monkeypatch.delenv("FEATURE_ORDER_WITHOUT_TOKEN", raising=False)
    monkeypatch.delenv("FEATURE_RESOLVER_LEARN_FROM_QUOTES", raising=False)
    yield
    settings_module.get_settings.cache_clear()


def test_plan_allows_missing_token_with_exchange_and_tradingsymbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATURE_ORDER_WITHOUT_TOKEN", "true")
    resolver = DummyResolver()
    mdm = DummyMDM(resolver, tick={"ltp": 100.0})
    broker = DummyBroker({})
    manager = UnifiedManager(broker=broker, mdm=mdm, ws=None, risk=None)
    plan = manager.plan("NIFTY25OCT25900CE", "BUY", 1)
    assert plan.ok
    assert plan.est_required > 0
    assert plan.qty >= 1


def test_get_price_learns_instrument_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FEATURE_RESOLVER_LEARN_FROM_QUOTES", "true")
    resolver = DummyResolver()
    mdm = DummyMDM(resolver, tick=None)
    quote_payload = {
        "last_price": 100.0,
        "instrument_token": 123456,
        "tradingsymbol": "NIFTY25OCT25900CE",
        "exchange": "NFO",
    }
    broker = DummyBroker(quote_payload)
    manager = UnifiedManager(broker=broker, mdm=mdm, ws=None, risk=None)
    assert resolver.resolve_symbol_to_token("NIFTY25OCT25900CE") is None
    price = manager.get_price("NIFTY25OCT25900CE")
    assert price.ltp == pytest.approx(100.0)
    learned = resolver.resolve_symbol_to_token("NIFTY25OCT25900CE")
    assert learned == 123456
