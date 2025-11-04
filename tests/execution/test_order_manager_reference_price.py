from pathlib import Path

import pytest

from nifty_scalper_bot.execution.order_manager import (
    OrderManager,
    resolve_reference_price,
)
from nifty_scalper_bot.utils.errors import OrderPlacementError


class _RefreshingMDM:
    def __init__(self, *, refresh_success: bool) -> None:
        self._quote: dict[str, float] = {}
        self._age_ms: int = 1_000_000_000
        self.refresh_success = refresh_success
        self.refresh_calls = 0

    def get_quote(self, symbol: str) -> dict[str, float]:  # noqa: D401 - stub
        return dict(self._quote)

    def quote_age_ms(self, symbol: str) -> int:  # noqa: D401 - stub
        return self._age_ms

    def cached_mid(
        self, symbol: str
    ) -> tuple[float | None, int | None]:  # noqa: D401 - stub
        return None, None

    def refresh_quote_now(
        self, symbol: str
    ) -> dict[str, float] | None:  # noqa: D401 - stub
        self.refresh_calls += 1
        if self.refresh_success:
            self._quote = {"ltp": 205.0}
            self._age_ms = 25
        else:
            self._quote = {}
            self._age_ms = 1_000_000_000
        return dict(self._quote)


@pytest.mark.parametrize("refresh_success", [True, False])
def test_resolve_reference_price_refreshes_once(refresh_success: bool) -> None:
    mdm = _RefreshingMDM(refresh_success=refresh_success)
    price, meta = resolve_reference_price(
        "NIFTY",
        mdm=mdm,  # type: ignore[arg-type]
        require_depth=False,
        max_age_ms=2_000,
        allow_ltp_fallback=True,
        allow_market_protect=False,
        protect_slippage_bps=12,
    )
    assert mdm.refresh_calls == 1
    if refresh_success:
        assert price == pytest.approx(205.0)
        assert meta.source == "ltp"
        assert meta.age_ms is not None and meta.age_ms <= 2_000
    else:
        assert price is None
        assert meta.source == "stale_mid"
        assert meta.age_ms == 1_000_000_000


class _QuoteLessMDM:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def has_quote(self, symbol: str) -> bool:  # noqa: D401 - test stub
        return False

    def refresh_quote_now(
        self, symbol: str, trace_id: str | None = None
    ) -> dict[str, float] | None:  # noqa: D401 - test stub
        self.refresh_calls += 1
        return None


class _StubBroker:
    pass


class _StubPositionManager:
    pass


class _StubLimiter:
    def acquire(self) -> None:  # noqa: D401 - test stub
        return None


class _TestOrderManager(OrderManager):
    def __init__(self, history_path: Path) -> None:
        super().__init__(
            _StubBroker(),
            _StubPositionManager(),
            _StubLimiter(),
            history_path,
        )


def test_no_quote_data_reason(tmp_path: Path) -> None:
    manager = _TestOrderManager(history_path=tmp_path / "orders.json")
    mdm = _QuoteLessMDM()

    with pytest.raises(OrderPlacementError) as excinfo:
        manager._handle_missing_reference_price(mdm, "NIFTY25O2025650CE")

    assert str(excinfo.value) == "NO_QUOTE_DATA"
    assert manager._last_skip_reason == "no_quote_data"
