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


class _BasketInstrumentManager:
    def __init__(self) -> None:
        self._token_to_symbol = {
            256265: "NSE:NIFTY",
            999001: "NFO:NIFTY26MAYFUT",
            1101: "NFO:NIFTY26MAY24850CE",
            1102: "NFO:NIFTY26MAY24900CE",
            1103: "NFO:NIFTY26MAY24950CE",
            1104: "NFO:NIFTY26MAY25000CE",
            1105: "NFO:NIFTY26MAY25050CE",
            1106: "NFO:NIFTY26MAY25100CE",
            1107: "NFO:NIFTY26MAY25150CE",
            2101: "NFO:NIFTY26MAY24850PE",
            2102: "NFO:NIFTY26MAY24900PE",
            2103: "NFO:NIFTY26MAY24950PE",
            2104: "NFO:NIFTY26MAY25000PE",
            2105: "NFO:NIFTY26MAY25050PE",
            2106: "NFO:NIFTY26MAY25100PE",
            2107: "NFO:NIFTY26MAY25150PE",
        }

    def is_loaded(self) -> bool:
        return True

    def select_tokens_for_universe(
        self,
        *,
        base: str,
        spot_price: float,
        strikes_around_atm: int,
        strike_step: int,
    ) -> list[int]:
        assert base == "NIFTY"
        assert strikes_around_atm == 3
        assert strike_step == 50
        return [1101, 1102, 1103, 1104, 1105, 1106, 1107, 2101, 2102, 2103, 2104, 2105, 2106, 2107]

    def get_symbol(self, token: int) -> str | None:
        return self._token_to_symbol.get(token)

    def get_token(self, symbol: str) -> int:
        if symbol == "NSE:NIFTY":
            raise RuntimeError("spot must not be resolved via instrument manager")
        for token, value in self._token_to_symbol.items():
            if value == symbol:
                return token
        raise RuntimeError(symbol)


def test_build_canonical_active_basket_is_limited_to_spot_futures_atm_pm3() -> None:
    manager = _BasketInstrumentManager()
    basket = app._build_canonical_active_basket(
        instrument_manager=manager,
        spot_token_resolver=lambda symbol: 256265 if symbol == "NSE:NIFTY" else 0,
        spot_ltp=25010.0,
        futures_symbol="NFO:NIFTY26MAYFUT",
        strike_step=50,
        strikes_around_atm=3,
    )
    assert basket["spot_symbol"] == "NSE:NIFTY"
    assert basket["futures_symbol"] == "NFO:NIFTY26MAYFUT"
    assert len(basket["ce_symbols"]) == 7
    assert len(basket["pe_symbols"]) == 7
    assert len(basket["symbols"]) == 16


def test_hydration_tracking_uses_active_basket_not_option_universe(monkeypatch) -> None:
    from nifty_scalper_bot.core.option_universe import OptionUniverseConfig, OptionUniverseManager

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setattr(
        OptionUniverseManager,
        "_build_universe",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("manual generation should not run")),
    )
    universe = OptionUniverseManager(OptionUniverseConfig())
    basket = {
        "selected_ce": "NFO:NIFTY26JUN25000CE",
        "selected_pe": "NFO:NIFTY26JUN25000PE",
        "option_symbols": ["NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"],
    }

    symbols = app._get_symbols(app.AppConfig(), option_universe=universe, active_contract_basket=basket)

    assert symbols == ["NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"]


def test_get_symbols_uses_ce_pe_symbol_fallback_without_option_symbols(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    basket = {
        "ce_symbols": ["NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25050CE"],
        "pe_symbols": ["NFO:NIFTY26JUN25000PE", "NFO:NIFTY26JUN25050PE"],
        "selected_ce": "NFO:NIFTY26JUN99999CE",
        "selected_pe": "NFO:NIFTY26JUN99999PE",
    }

    symbols = app._get_symbols(app.AppConfig(), active_contract_basket=basket)

    assert symbols == [
        "NFO:NIFTY26JUN25000CE",
        "NFO:NIFTY26JUN25050CE",
        "NFO:NIFTY26JUN25000PE",
        "NFO:NIFTY26JUN25050PE",
    ]
