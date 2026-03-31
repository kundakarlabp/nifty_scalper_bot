from __future__ import annotations

from typing import Any, Mapping

import pytest

import nifty_scalper_bot.config.settings as app_settings
from nifty_scalper_bot.execution.order_manager import (
    RefPriceMeta,
    resolve_reference_price,
)
from nifty_scalper_bot.utils.errors import OrderPlacementError


class DummyMDM:
    def __init__(
        self,
        *,
        quote: Mapping[str, Any] | None,
        age_ms: int,
        cached_mid: tuple[float | None, int | None] = (None, None),
    ) -> None:
        self._quote = quote
        self._age_ms = age_ms
        self._cached_mid = cached_mid

    def get_quote(self, symbol: str) -> Mapping[str, Any] | None:  # noqa: D401 - stub
        return self._quote

    def quote_age_ms(self, symbol: str) -> int:  # noqa: D401 - stub
        return int(self._age_ms)

    def cached_mid(
        self, symbol: str
    ) -> tuple[float | None, int | None]:  # noqa: D401 - stub
        return self._cached_mid


def test_reference_price_ladder_prefers_ltp_when_depth_missing() -> None:
    mdm = DummyMDM(
        quote={"ltp": 201.25},
        age_ms=500,
        cached_mid=(None, None),
    )
    price, meta = resolve_reference_price(
        "NFO:NIFTY24OCT20000CE",
        mdm=mdm,
        require_depth=True,
        max_age_ms=app_settings.ORDER_MAX_QUOTE_AGE_MS,
        allow_ltp_fallback=True,
        allow_market_protect=False,
        protect_slippage_bps=app_settings.ORDER_PROTECT_SLIPPAGE_BPS,
    )
    assert price == pytest.approx(201.25)
    assert isinstance(meta, RefPriceMeta)
    assert meta.source == "ltp"
    assert meta.market_protect is False


def test_reference_price_ladder_uses_cached_mid_when_ltp_stale() -> None:
    mdm = DummyMDM(
        quote={"ltp": 0},
        age_ms=5_000,
        cached_mid=(120.5, 1_000),
    )
    price, meta = resolve_reference_price(
        "NFO:NIFTY24OCT20000CE",
        mdm=mdm,
        require_depth=True,
        max_age_ms=app_settings.ORDER_MAX_QUOTE_AGE_MS,
        allow_ltp_fallback=True,
        allow_market_protect=False,
        protect_slippage_bps=app_settings.ORDER_PROTECT_SLIPPAGE_BPS,
    )
    assert price == pytest.approx(120.5)
    assert meta.source == "stale_mid"
    assert meta.market_protect is False


def test_reference_price_short_circuits_when_ladder_fails() -> None:
    mdm = DummyMDM(quote={}, age_ms=10_000, cached_mid=(None, None))
    price, meta = resolve_reference_price(
        "NFO:NIFTY24OCT20000CE",
        mdm=mdm,
        require_depth=True,
        max_age_ms=app_settings.ORDER_MAX_QUOTE_AGE_MS,
        allow_ltp_fallback=False,
        allow_market_protect=False,
        protect_slippage_bps=app_settings.ORDER_PROTECT_SLIPPAGE_BPS,
    )
    assert price is None
    assert meta.market_protect is False

    with pytest.raises(OrderPlacementError):
        if price is None and not meta.market_protect:
            raise OrderPlacementError("NO_REFERENCE_PRICE")


def test_reference_price_allows_market_protect_when_enabled() -> None:
    mdm = DummyMDM(quote={}, age_ms=10_000, cached_mid=(None, None))
    price, meta = resolve_reference_price(
        "NFO:NIFTY24OCT20000CE",
        mdm=mdm,
        require_depth=True,
        max_age_ms=app_settings.ORDER_MAX_QUOTE_AGE_MS,
        allow_ltp_fallback=False,
        allow_market_protect=True,
        protect_slippage_bps=app_settings.ORDER_PROTECT_SLIPPAGE_BPS,
    )
    assert price is None
    assert meta.market_protect is True
