from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution.entry_recovery import install_entry_recovery
from nifty_scalper_bot.execution.order_manager import TradePlan, TradePlanSubmitResult


class _Logger:
    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def critical(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Quotes:
    def __init__(self, quote: dict[str, Any]) -> None:
        self.quote = quote

    def refresh_quote_now(self, _symbol: str, **_kwargs: Any) -> None:
        return None

    def get_quote(self, _symbol: str) -> dict[str, Any]:
        return dict(self.quote)


class _Broker:
    def get_orders(self) -> list[dict[str, Any]]:
        return []

    def get_positions(self) -> list[dict[str, Any]]:
        return []


class _Resolver:
    def __init__(self, *, lot: int | None = None, freeze: int | None = None) -> None:
        self.lot = lot
        self.freeze = freeze

    def get_lot_size(self, _symbol: str) -> int | None:
        return self.lot

    def get_freeze_quantity(self, _symbol: str) -> int | None:
        return self.freeze


class _Manager:
    def __init__(
        self,
        responses: list[TradePlanSubmitResult],
        *,
        quote: dict[str, Any] | None = None,
        resolver: Any | None = None,
        margin: float = 20_000.0,
        live: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.plans: list[TradePlan] = []
        self._logger = _Logger()
        self._market_data = _Quotes(quote or {"ask": 100.0, "bid": 99.5, "ltp": 99.8})
        self._broker = _Broker()
        self._instrument_resolver = resolver
        self._margin_factor = 1.0
        self._margin_buffer = 0.95
        self._last_order_decision: dict[str, Any] = {}
        self.margin = margin
        self.live = live

    def submit_trade_plan_result(self, plan: TradePlan) -> TradePlanSubmitResult:
        self.plans.append(plan)
        return self.responses.pop(0)

    def _resolve_available_margin(self) -> tuple[float, str]:
        return self.margin, "fresh"

    def is_live_mode(self) -> bool:
        return bool(self.live)


install_entry_recovery(_Manager)


def _plan(quantity: int = 130, side: str = "BUY") -> TradePlan:
    return TradePlan(
        symbol="NFO:NIFTY2662324050PE",
        side=side,  # type: ignore[arg-type]
        quantity=quantity,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        signal_id="signal-hardening",
        trace_id="trace-hardening",
        tag="runner-entry",
    )


def _reject(message: str) -> TradePlanSubmitResult:
    return TradePlanSubmitResult(
        accepted=False,
        reason="broker_rejected",
        details={"broker_message": message},
        broker_attempted=True,
    )


def _accept(order_id: str = "retry-order") -> TradePlanSubmitResult:
    return TradePlanSubmitResult(
        accepted=True,
        order_id=order_id,
        reason="accepted",
        broker_attempted=True,
    )


def test_margin_resize_uses_nifty_lot_fallback_not_single_unit(monkeypatch) -> None:
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    monkeypatch.delenv("NIFTY_LOT_SIZE", raising=False)
    monkeypatch.delenv("INSTRUMENTS__NIFTY_LOT_SIZE", raising=False)
    manager = _Manager(
        [
            _reject("insufficient funds; required margin exceeds available margin"),
            _accept(),
        ],
        margin=10_000.0,
    )

    result = manager.submit_trade_plan_result(_plan(quantity=130))

    assert result.accepted is True
    assert len(manager.plans) == 2
    assert manager.plans[1].quantity == 65
    assert result.details["entry_recovery"]["retry_quantity"] == 65


def test_margin_resize_prefers_instrument_resolver_lot_size() -> None:
    manager = _Manager(
        [
            _reject("insufficient funds; required margin exceeds available margin"),
            _accept(),
        ],
        resolver=_Resolver(lot=50),
        margin=8_000.0,
    )

    result = manager.submit_trade_plan_result(_plan(quantity=100))

    assert result.accepted is True
    assert manager.plans[1].quantity == 50
    assert result.details["entry_recovery"]["retry_quantity"] == 50


def test_freeze_limit_recovery_caps_to_whole_nifty_lot(monkeypatch) -> None:
    monkeypatch.delenv("ORDER_FREEZE_QUANTITY", raising=False)
    manager = _Manager(
        [_reject("maximum order quantity freeze limit exceeded"), _accept()],
        resolver=_Resolver(lot=None, freeze=120),
    )

    result = manager.submit_trade_plan_result(_plan(quantity=130))

    assert result.accepted is True
    assert len(manager.plans) == 2
    assert manager.plans[1].quantity == 65
    assert result.details["entry_recovery"]["retry_quantity"] == 65


def test_price_recovery_blocks_excessive_reprice_before_duplicate_entry(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENTRY_RECOVERY_MAX_REPRICE_DEVIATION_PCT", "3.0")
    manager = _Manager(
        [
            _reject("invalid limit price: price out of range"),
            _accept("should-not-submit"),
        ],
        quote={"ask": 145.0, "bid": 144.5, "ltp": 144.8},
    )

    result = manager.submit_trade_plan_result(_plan(quantity=65))

    assert result.accepted is False
    assert len(manager.plans) == 1
    recovery = result.details["entry_recovery"]
    assert recovery["outcome"] == "rebuild_failed"
    assert recovery["reprice_deviation_pct"] == 45.0
    assert recovery["max_reprice_deviation_pct"] == 3.0


def test_live_price_recovery_blocks_ltp_only_quote_before_retry(monkeypatch) -> None:
    monkeypatch.delenv("ENTRY_RECOVERY_MAX_REPRICE_DEVIATION_PCT", raising=False)
    manager = _Manager(
        [
            _reject("invalid limit price: price out of range"),
            _accept("should-not-submit"),
        ],
        quote={"ltp": 101.0, "last_price": 101.0},
        live=True,
    )

    result = manager.submit_trade_plan_result(_plan(quantity=65))

    assert result.accepted is False
    assert len(manager.plans) == 1
    recovery = result.details["entry_recovery"]
    assert recovery["outcome"] == "live_bid_ask_unavailable"


def test_live_price_recovery_rebuilds_from_ask_not_ltp(monkeypatch) -> None:
    monkeypatch.delenv("ENTRY_RECOVERY_MAX_REPRICE_DEVIATION_PCT", raising=False)
    manager = _Manager(
        [_reject("invalid limit price: price out of range"), _accept("live-retry")],
        quote={"bid": 100.5, "ask": 101.0, "ltp": 98.0},
        live=True,
    )

    result = manager.submit_trade_plan_result(_plan(quantity=65))

    assert result.accepted is True
    assert len(manager.plans) == 2
    assert manager.plans[1].entry_price == 101.0
    assert manager.plans[1].stop_loss == 91.0
    assert manager.plans[1].take_profit == 121.0
    assert result.details["entry_recovery"]["retry_entry"] == 101.0


def test_shadow_price_recovery_can_still_use_ltp_fallback() -> None:
    manager = _Manager(
        [_reject("invalid limit price: price out of range"), _accept("shadow-retry")],
        quote={"ltp": 101.0},
        live=False,
    )

    result = manager.submit_trade_plan_result(_plan(quantity=65))

    assert result.accepted is True
    assert len(manager.plans) == 2
    assert manager.plans[1].entry_price == 101.0
    assert result.details["entry_recovery"]["retry_entry"] == 101.0


def test_recovery_retry_cannot_exceed_first_gate_approved_quantity(
    monkeypatch,
) -> None:
    """A retry may reduce further but must never restore gated-away quantity.

    Original request 130 (2 lots). The entry margin gate approved 65 on the
    first submission. Even when a later margin/price refresh would permit 130
    again, recovery must rebuild at most the first approved 65.
    """
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    first = _reject("margin required 20000 available 5000")
    # The first submission carries the frozen sizing record.
    first.details = {
        **(first.details or {}),
        "entry_sizing_requested_quantity": 130,
        "entry_sizing_effective_quantity": 65,
        "entry_sizing_lot_size": 65,
    }
    # Plenty of margin at retry time: an ungated resize would ask for 130.
    manager = _Manager(
        [first, _accept()],
        resolver=_Resolver(lot=65),
        margin=10_000_000.0,
    )

    result = manager.submit_trade_plan_result(_plan(quantity=130))

    assert result.accepted
    # Exactly two submissions: the original and one retry.
    assert len(manager.plans) == 2
    assert manager.plans[0].quantity == 130
    retry_plan = manager.plans[1]
    assert retry_plan.quantity <= 65
    assert retry_plan.quantity != 130
    assert retry_plan.quantity % 65 == 0
    assert retry_plan.requested_lots == retry_plan.quantity // 65
    assert retry_plan.resolved_lot_size == 65
    assert result.details.get("entry_recovery") is not None


def test_recovery_without_sizing_record_does_not_invent_a_cap(monkeypatch) -> None:
    """Missing/untrusted sizing record must not silently raise the quantity."""
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    manager = _Manager(
        [_reject("margin required 20000 available 5000"), _accept()],
        resolver=_Resolver(lot=65),
        margin=10_000_000.0,
    )

    manager.submit_trade_plan_result(_plan(quantity=130))

    retry_plan = manager.plans[1]
    # No frozen record exists, so the normal resize path applies; it must
    # still never exceed the caller's original request.
    assert retry_plan.quantity <= 130
    assert retry_plan.quantity % 65 == 0
