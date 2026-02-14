from types import SimpleNamespace

from nifty_scalper_bot.core import app


class _UniverseStub:
    def __init__(self) -> None:
        self.underlying: float | None = None

    def update_underlying(self, ltp: float) -> None:
        self.underlying = ltp

    def get_filtered_universe(self, _ltp: float) -> list[str]:
        return ['NFO:NIFTY26FEB25000CE']


class _BrokerNoSpot:
    def ltp(self, _symbols: list[str]) -> dict[str, dict[str, float]]:
        return {'NIFTY 50': {'last_price': 25300.0}}


class _BrokerWithSpot:
    def ltp(self, _symbols: list[str]) -> dict[str, dict[str, float]]:
        return {'NSE:NIFTY 50': {'last_price': 25382.4}}


def test_get_symbols_aborts_when_live_spot_symbol_missing() -> None:
    cfg = SimpleNamespace(symbols=None)
    universe = _UniverseStub()

    symbols = app._get_symbols(cfg, broker=_BrokerNoSpot(), option_universe=universe)

    assert symbols == []
    assert universe.underlying is None


def test_get_symbols_uses_nse_nifty_50_symbol() -> None:
    cfg = SimpleNamespace(symbols=None)
    universe = _UniverseStub()

    symbols = app._get_symbols(cfg, broker=_BrokerWithSpot(), option_universe=universe)

    assert symbols == ['NFO:NIFTY26FEB25000CE']
    assert universe.underlying == 25382.4
