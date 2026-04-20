from types import SimpleNamespace

from nifty_scalper_bot.core import app


class _UniverseStub:
    def __init__(self) -> None:
        self.underlying: float | None = None

    def update_underlying(self, ltp: float) -> None:
        self.underlying = ltp

    def get_filtered_universe(self, _ltp: float) -> list[str]:
        return ["NFO:NIFTY26FEB25000CE"]


class _BrokerNoSpot:
    def ltp(self, _symbols: list[str]) -> dict[str, dict[str, float]]:
        return {"NIFTY 50": {"last_price": 25300.0}}


class _BrokerWithSpot:
    def ltp(self, _symbols: list[str]) -> dict[str, dict[str, float]]:
        return {"NSE:NIFTY": {"last_price": 25382.4}}


class _ResolverStub:
    def resolve(self, symbol: str) -> int | None:
        if symbol == "NFO:NIFTY26FEB25000CE":
            return None
        if symbol == "NFO:NIFTY26FEB25050CE":
            return 123456
        return None

    def option_contracts(self, base: str) -> list[dict[str, object]]:
        if base != "NIFTY":
            return []
        return [
            {
                "tradingsymbol": "NIFTY26FEB25050CE",
                "strike": 25050.0,
                "expiry": "2026-02-26",
            }
        ]


def test_get_symbols_accepts_alternate_spot_symbol_keys() -> None:
    cfg = SimpleNamespace(symbols=None)
    universe = _UniverseStub()

    symbols = app._get_symbols(cfg, broker=_BrokerNoSpot(), option_universe=universe)

    assert symbols == ["NFO:NIFTY26FEB25000CE"]
    assert universe.underlying == 25300.0


def test_get_symbols_uses_nse_nifty_50_symbol() -> None:
    cfg = SimpleNamespace(symbols=None)
    universe = _UniverseStub()

    symbols = app._get_symbols(cfg, broker=_BrokerWithSpot(), option_universe=universe)

    assert symbols == ["NFO:NIFTY26FEB25000CE"]
    assert universe.underlying == 25382.4


def test_get_symbols_remaps_unresolved_option_to_available_contract() -> None:
    cfg = SimpleNamespace(symbols=None)
    universe = _UniverseStub()

    symbols = app._get_symbols(
        cfg,
        broker=_BrokerWithSpot(),
        option_universe=universe,
        resolver=_ResolverStub(),
    )

    assert symbols == ["NFO:NIFTY26FEB25050CE"]


def test_get_current_nifty_futures_symbol_has_no_logging_side_effect(
    caplog,
) -> None:
    caplog.set_level("INFO")

    symbol = app._get_current_nifty_futures_symbol()

    assert symbol.startswith("NFO:NIFTY")
    assert symbol.endswith("FUT")
    assert all("Using futures symbol" not in rec.message for rec in caplog.records)
