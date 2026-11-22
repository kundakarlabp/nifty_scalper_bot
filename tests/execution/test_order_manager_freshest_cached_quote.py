from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.execution.order_manager import OrderManager, TradePlan


class _TickProvider:
    def __init__(self, quote: dict[str, object]) -> None:
        self.quote = dict(quote)

    def get_latest_tick(self, _symbol: str) -> dict[str, object]:
        return dict(self.quote)


class _PullTrackingDataHub(_TickProvider):
    def __init__(self) -> None:
        super().__init__({})
        self.pull_attempts = 0

    def get_quote(
        self,
        _symbol: str,
        allow_pull: bool = True,
    ) -> dict[str, object] | None:
        if allow_pull:
            self.pull_attempts += 1
        return None


def _manager(*, data_hub: _TickProvider, mdm: _TickProvider) -> OrderManager:
    manager = object.__new__(OrderManager)
    manager._data_hub = data_hub
    manager.data_hub = None
    manager._market_data = None
    manager._market_data_manager = mdm
    manager.market_data_manager = None
    return manager


def _quote(
    symbol: str,
    *,
    marker: str,
    age_s: float,
    executable: bool = True,
) -> dict[str, object]:
    quote: dict[str, object] = {
        "symbol": symbol,
        "ltp": 100.0,
        "received_at": time.time() - age_s,
        "marker": marker,
    }
    if executable:
        quote.update(
            {
                "bid": 99.90,
                "ask": 100.10,
                "bid_quantity": 65,
                "ask_quantity": 65,
            }
        )
    return quote


def _depth_quote(
    symbol: str,
    *,
    marker: str = "ws_full_depth",
    bid_qty: int = 130,
    ask_qty: int = 130,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "ltp": 100.0,
        "received_at": time.time() - 0.02,
        "marker": marker,
        "depth_available": True,
        "depth": {
            "buy": [{"price": 99.90, "quantity": bid_qty}],
            "sell": [{"price": 100.10, "quantity": ask_qty}],
        },
    }


def test_stale_exchange_book_is_not_freshened_by_recent_receipt() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    quote = _depth_quote(symbol)
    quote["exchange_timestamp"] = datetime.now(timezone.utc) - timedelta(seconds=45)
    quote["timestamp_quality"] = "exchange"
    manager = _manager(data_hub=_TickProvider(quote), mdm=_TickProvider(quote))

    diagnostics = manager._extract_quote_diagnostics(quote)
    assert diagnostics["timestamp_key"] == "exchange_timestamp"
    assert diagnostics["age_ms"] > 40_000


def test_future_exchange_book_is_unusable_even_with_recent_receipt() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    quote = _depth_quote(symbol)
    quote["exchange_timestamp"] = datetime.now(timezone.utc) + timedelta(seconds=60)
    manager = _manager(data_hub=_TickProvider(quote), mdm=_TickProvider(quote))
    diagnostics = manager._extract_quote_diagnostics(quote)
    plan = TradePlan(
        symbol=symbol,
        side="BUY",
        quantity=65,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
    )

    assert diagnostics["age_ms"] is None
    rejection = manager._validate_live_entry_quote(plan, diagnostics)
    assert rejection is not None
    assert rejection.reason == "entry_quote_timestamp_unusable"


def test_explicit_invalid_or_receipt_only_source_cannot_authorize_live_book() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(data_hub=_TickProvider({}), mdm=_TickProvider({}))
    for provenance in (
        {"timestamp_quality": "synthetic"},
        {"source_timestamp_valid": False},
        {"timestamp_source": "received_at", "source_timestamp_valid": True},
    ):
        quote = {**_depth_quote(symbol), **provenance}
        diagnostics = manager._extract_quote_diagnostics(quote)
        assert diagnostics["age_ms"] is None


def test_legacy_receipt_only_quote_keeps_nonlive_age_compatibility() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    quote = _depth_quote(symbol)
    manager = _manager(data_hub=_TickProvider(quote), mdm=_TickProvider(quote))

    assert manager._extract_quote_diagnostics(quote)["age_ms"] is not None


def test_runtime_owns_quote_methods_without_facade_monkey_patching() -> None:
    from nifty_scalper_bot.execution import order_manager_core
    from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager

    assert OrderManager is RuntimeOrderManager
    for name in ("_extract_quote_diagnostics", "_get_latest_quote_safe"):
        method = RuntimeOrderManager.__dict__[name]
        assert method.__module__ == "nifty_scalper_bot.execution.runtime_order_manager"
        assert method is not getattr(order_manager_core.OrderManager, name)


def test_direct_runtime_quote_preflight_uses_cached_full_book_without_pull() -> None:
    from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager

    symbol = "NFO:NIFTY26AUG24050PE"
    data_hub = _PullTrackingDataHub()
    manager = object.__new__(RuntimeOrderManager)
    manager._data_hub = data_hub
    manager.data_hub = None
    manager._market_data = None
    manager._market_data_manager = _TickProvider(
        _depth_quote(symbol, bid_qty=130, ask_qty=195)
    )
    manager.market_data_manager = None

    selected = manager._get_latest_quote_safe(symbol)
    assert selected is not None
    assert selected["depth"]["sell"][0]["quantity"] == 195
    diagnostics = manager._extract_quote_diagnostics(selected)
    assert diagnostics["bid"] == 99.90
    assert diagnostics["ask"] == 100.10
    assert diagnostics["bid_qty"] == 130
    assert diagnostics["ask_qty"] == 195
    assert data_hub.pull_attempts == 0


def test_order_preflight_uses_freshest_cached_provider_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_quote(symbol, marker="processed", age_s=2.4)),
        mdm=_TickProvider(_quote(symbol, marker="ws_fast", age_s=0.03)),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "ws_fast"


def test_order_preflight_keeps_first_quote_when_it_is_already_freshest() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_quote(symbol, marker="processed", age_s=0.02)),
        mdm=_TickProvider(_quote(symbol, marker="ws_fast", age_s=0.20)),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "processed"


def test_order_preflight_preserves_older_executable_quote_over_fresher_ltp() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_quote(symbol, marker="executable", age_s=0.20)),
        mdm=_TickProvider(
            _quote(symbol, marker="ltp_only", age_s=0.01, executable=False)
        ),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "executable"


def test_order_preflight_upgrades_ltp_quote_to_executable_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(
            _quote(symbol, marker="ltp_only", age_s=0.01, executable=False)
        ),
        mdm=_TickProvider(_quote(symbol, marker="executable", age_s=0.20)),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "executable"


def test_order_preflight_keeps_freshest_quote_when_only_ltp_is_available() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(
            _quote(symbol, marker="processed", age_s=0.20, executable=False)
        ),
        mdm=_TickProvider(
            _quote(symbol, marker="ws_fast", age_s=0.01, executable=False)
        ),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "ws_fast"


def test_order_preflight_extracts_zerodha_depth_as_executable_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_depth_quote(symbol)),
        mdm=_TickProvider(_depth_quote(symbol)),
    )

    diagnostics = manager._extract_quote_diagnostics(_depth_quote(symbol))

    assert diagnostics["bid"] == 99.90
    assert diagnostics["ask"] == 100.10
    assert diagnostics["bid_qty"] == 130
    assert diagnostics["ask_qty"] == 130
    assert diagnostics["depth_qty"] == 260
    assert diagnostics["spread_pct"] > 0


def test_live_entry_accepts_complete_zerodha_depth_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_depth_quote(symbol)),
        mdm=_TickProvider(_depth_quote(symbol)),
    )
    plan = TradePlan(
        symbol=symbol,
        side="BUY",
        quantity=65,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        max_spread_pct=5.0,
    )

    diagnostics = manager._extract_quote_diagnostics(_depth_quote(symbol))
    rejection = manager._validate_live_entry_quote(plan, diagnostics)

    assert rejection is None


def test_live_entry_keeps_depth_quantity_guard_for_zerodha_depth_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    quote = _depth_quote(symbol, ask_qty=25)
    manager = _manager(data_hub=_TickProvider(quote), mdm=_TickProvider(quote))
    plan = TradePlan(
        symbol=symbol,
        side="BUY",
        quantity=65,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        max_spread_pct=5.0,
    )

    diagnostics = manager._extract_quote_diagnostics(quote)
    rejection = manager._validate_live_entry_quote(plan, diagnostics)

    assert rejection is not None
    assert rejection.reason == "entry_executable_depth_insufficient"


def test_order_preflight_does_not_trigger_http_pull_before_cached_ws_lookup() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    data_hub = _PullTrackingDataHub()
    manager = _manager(
        data_hub=data_hub,
        mdm=_TickProvider(_depth_quote(symbol, marker="mdm_ws_full")),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "mdm_ws_full"
    assert data_hub.pull_attempts == 0


def test_execution_depth_helpers_use_cached_ws_full_quote_only() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    data_hub = _PullTrackingDataHub()
    manager = _manager(
        data_hub=data_hub,
        mdm=_TickProvider(_depth_quote(symbol, bid_qty=130, ask_qty=195)),
    )

    class _Logger:
        def debug(self, *_args, **_kwargs) -> None:
            return None

        def info(self, *_args, **_kwargs) -> None:
            return None

        def error(self, *_args, **_kwargs) -> None:
            return None

    manager._logger = _Logger()
    manager._history = []

    depth = manager.get_best_bid_ask_depth(symbol)
    queue = manager.calculate_queue_position(symbol, "BUY", 100.10)
    average, slippage, historical = manager._estimate_order_slippage(symbol, "BUY", 65)

    assert depth["bid"] == 99.90
    assert depth["ask"] == 100.10
    assert depth["bid_size"] == 130.0
    assert depth["ask_size"] == 195.0
    assert queue >= 195
    assert average == 100.10
    assert slippage == 0.0
    assert historical is None
    assert data_hub.pull_attempts == 0
