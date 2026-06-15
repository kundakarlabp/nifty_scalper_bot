from nifty_scalper_bot.core.app import _build_canonical_active_basket


class _FakeInstrumentManager:
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
        assert spot_price > 0
        return list(range(10, 24))

    def get_symbol(self, token: int) -> str:
        symbols = {
            10: "NFO:NIFTY26APR25000CE",
            11: "NFO:NIFTY26APR25050CE",
            12: "NFO:NIFTY26APR25100CE",
            13: "NFO:NIFTY26APR25150CE",
            14: "NFO:NIFTY26APR25200CE",
            15: "NFO:NIFTY26APR25250CE",
            16: "NFO:NIFTY26APR25300CE",
            17: "NFO:NIFTY26APR25000PE",
            18: "NFO:NIFTY26APR25050PE",
            19: "NFO:NIFTY26APR25100PE",
            20: "NFO:NIFTY26APR25150PE",
            21: "NFO:NIFTY26APR25200PE",
            22: "NFO:NIFTY26APR25250PE",
            23: "NFO:NIFTY26APR25300PE",
        }
        return symbols[token]

    def get_token(self, symbol: str) -> int:
        assert symbol == "NFO:NIFTY26APRFUT"
        return 123456


def test_build_canonical_active_basket_includes_spot_futures_and_options() -> None:
    basket = _build_canonical_active_basket(
        instrument_manager=_FakeInstrumentManager(),
        spot_token_resolver=lambda _symbol: 256265,
        spot_ltp=25125,
        futures_symbol="NFO:NIFTY26APRFUT",
        strike_step=50,
        strikes_around_atm=3,
    )

    assert basket["spot_symbol"] == "NSE:NIFTY"
    assert basket["futures_symbol"] == "NFO:NIFTY26APRFUT"
    assert basket["spot_token"] == 256265
    assert basket["futures_token"] == 123456
    assert len(basket["ce_symbols"]) == 7
    assert len(basket["pe_symbols"]) == 7


class _FakeIMWithOptions:
    """InstrumentManager whose get_token RAISES for unknown/empty symbols,
    mirroring the live FATAL 'token not found' behavior."""

    def is_loaded(self) -> bool:
        return True

    def select_tokens_for_universe(self, *, base, spot_price, strikes_around_atm, strike_step):
        return list(range(10, 24))

    def get_symbol(self, token: int) -> str:
        symbols = {
            10: "NFO:NIFTY26APR25000CE", 11: "NFO:NIFTY26APR25050CE",
            12: "NFO:NIFTY26APR25100CE", 13: "NFO:NIFTY26APR25150CE",
            14: "NFO:NIFTY26APR25200CE", 15: "NFO:NIFTY26APR25250CE",
            16: "NFO:NIFTY26APR25300CE", 17: "NFO:NIFTY26APR25000PE",
            18: "NFO:NIFTY26APR25050PE", 19: "NFO:NIFTY26APR25100PE",
            20: "NFO:NIFTY26APR25150PE", 21: "NFO:NIFTY26APR25200PE",
            22: "NFO:NIFTY26APR25250PE", 23: "NFO:NIFTY26APR25300PE",
        }
        return symbols[token]

    def get_token(self, symbol: str) -> int:
        key = str(symbol).strip()
        if not key:
            raise RuntimeError(f"[FATAL] InstrumentManager: token not found for '{symbol}'.")
        # deterministic fake token for any non-empty symbol
        return abs(hash(key)) % 1_000_000


async def test_empty_futures_symbol_does_not_abort_basket_build() -> None:
    # Regression: an unresolved (empty) active future previously caused
    # get_token('') to raise and abort the whole live basket build.
    basket = _build_canonical_active_basket(
        instrument_manager=_FakeIMWithOptions(),
        spot_token_resolver=lambda _s: 256265,
        spot_ltp=25125,
        futures_symbol="",  # unresolved future
        strike_step=50,
        strikes_around_atm=3,
    )
    assert basket["spot_symbol"] == "NSE:NIFTY"
    assert basket["futures_symbol"] == ""
    assert basket["futures_token"] is None
    assert len(basket["ce_symbols"]) == 7
    assert len(basket["pe_symbols"]) == 7
    # empty futures must NOT leak into the symbol lists
    assert "" not in basket["symbols"]
    assert "" not in basket["all_symbols"]


async def test_unresolvable_futures_symbol_is_demoted_not_fatal() -> None:
    # A futures symbol that the instrument manager cannot resolve must be
    # demoted to "no futures context", not abort the build.
    class _IM(_FakeIMWithOptions):
        def get_token(self, symbol: str) -> int:
            if "FUT" in str(symbol):
                raise RuntimeError("token not found")
            return super().get_token(symbol)

    basket = _build_canonical_active_basket(
        instrument_manager=_IM(),
        spot_token_resolver=lambda _s: 256265,
        spot_ltp=25125,
        futures_symbol="NFO:NIFTY26JUNFUT",
        strike_step=50,
        strikes_around_atm=3,
    )
    assert basket["futures_symbol"] == ""
    assert basket["futures_token"] is None
    assert "" not in basket["all_symbols"]
