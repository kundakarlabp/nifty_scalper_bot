import pytest

from nifty_scalper_bot.execution.entry_price import EntryPriceModel
from nifty_scalper_bot.execution.order_manager import (
    RefPriceMeta,
    resolve_reference_price,
)


class _StubMarketDataManager:
    def __init__(
        self,
        quote: dict[str, float | None],
        age_ms: float,
        cached_mid: tuple[float | None, int | None] = (None, None),
    ) -> None:
        self._quote = quote
        self._age_ms = age_ms
        self._cached_mid = cached_mid

    def get_quote(self, symbol: str) -> dict[str, float | None]:
        return dict(self._quote)

    def quote_age_ms(self, symbol: str) -> float:
        return float(self._age_ms)

    def cached_mid(self, symbol: str) -> tuple[float | None, int | None]:
        return self._cached_mid


def test_buy_price_pct_spread() -> None:
    model = EntryPriceModel(tick=0.05, pct_of_spread=0.75)
    assert model.compute(side="BUY", bid=100.0, ask=101.0) == 100.75


def test_sell_price_pct_spread() -> None:
    model = EntryPriceModel(tick=0.05, pct_of_spread=0.75)
    assert model.compute(side="SELL", bid=100.0, ask=101.0) == 100.25


def test_resolve_reference_price_uses_ltp_when_depth_missing() -> None:
    mdm = _StubMarketDataManager({"ltp": 100.5}, 250.0)
    price, meta = resolve_reference_price(
        "NIFTY",
        require_depth=True,
        max_age_ms=2_000,
        allow_ltp_fallback=True,
        allow_market_protect=False,
        protect_slippage_bps=12,
        mdm=mdm,
    )
    assert price == pytest.approx(100.5)
    assert isinstance(meta, RefPriceMeta)
    assert meta.source == "ltp"
    assert meta.market_protect is False


def test_resolve_reference_price_returns_none_when_stale_and_no_protect() -> None:
    mdm = _StubMarketDataManager({}, 5_000.0)
    price, meta = resolve_reference_price(
        "NIFTY",
        require_depth=True,
        max_age_ms=1_000,
        allow_ltp_fallback=True,
        allow_market_protect=False,
        protect_slippage_bps=12,
        mdm=mdm,
    )
    assert price is None
    assert isinstance(meta, RefPriceMeta)
    assert meta.source == "stale_mid"
    assert meta.market_protect is False
