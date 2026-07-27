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
        "entry_sizing_symbol": "NFO:NIFTY2662324050PE",
        "entry_sizing_trace_id": "trace-hardening",
        "entry_sizing_signal_id": "signal-hardening",
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


class _PartialFillManager:
    """Manager exposing the real _update_from_response reconciliation path."""

    def __init__(self, *, resolver: Any | None = None) -> None:
        self._logger = _Logger()
        self._instrument_resolver = resolver
        self._broker = _Broker()
        self.blockers: list[str] = []

    def _update_from_response(self, order: Any, payload: dict[str, Any]) -> Any:
        return order

    def _latch_entry_blocker(self, *_a: Any, **_k: Any) -> None:
        return None


install_entry_recovery(_PartialFillManager)


class _EntryOrder:
    """Minimal OrderDetails-like entry order built from the EFFECTIVE plan."""

    def __init__(self, quantity: int, lot_size: int, requested_lots: int) -> None:
        self.order_id = "ORD-PARTIAL-1"
        self.symbol = "NFO:NIFTY2662324050PE"
        self.intent = "ENTRY"
        self.tag = "runner-entry"
        self.status = "PARTIALLY FILLED"
        self.quantity = quantity
        self.resolved_lot_size = lot_size
        self.requested_lots = requested_lots
        self.filled_quantity = 0


def test_partial_fill_lifecycle_uses_gate_reduced_quantity(monkeypatch) -> None:
    """Partial-fill reconciliation must use the EFFECTIVE 65, never the 130.

    Original request 130 (2 lots); the entry margin gate approved 65, so the
    broker order and therefore the OrderDetails carry 65. The stored lifecycle
    state must reflect 65 -- reconciliation must never wait on, or protect,
    a phantom remainder from the original 130.
    """
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    manager = _PartialFillManager(resolver=_Resolver(lot=65))
    order = _EntryOrder(quantity=65, lot_size=65, requested_lots=1)

    manager._update_from_response(
        order,
        {
            "status": "PARTIALLY FILLED",
            "filled_quantity": 20,
            "pending_quantity": 45,
        },
    )

    state = order.entry_lifecycle_state
    assert state["requested_quantity"] == 65
    assert state["requested_lots"] == 1
    assert state["resolved_lot_size"] == 65
    assert state["broker_filled_quantity"] == 20
    assert state["broker_pending_quantity"] == 45
    # The original oversized request must never appear anywhere in state.
    assert 130 not in [v for v in state.values() if isinstance(v, int)]
    # Any protective quantity derived from this state stays within 65.
    assert int(state.get("protected_quantity") or 0) <= 65


def test_partial_fill_lifecycle_lot_aligned_fill_tracks_protected_quantity(
    monkeypatch,
) -> None:
    """A lot-aligned partial fill records protected quantity within 65."""
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    manager = _PartialFillManager(resolver=_Resolver(lot=65))
    order = _EntryOrder(quantity=65, lot_size=65, requested_lots=1)

    manager._update_from_response(
        order,
        {
            "status": "COMPLETE",
            "filled_quantity": 65,
            "pending_quantity": 0,
        },
    )

    state = order.entry_lifecycle_state
    assert state["requested_quantity"] == 65
    assert state["requested_lots"] == 1
    assert state["broker_filled_quantity"] == 65
    assert int(state.get("protected_quantity") or 0) <= 65


def _sized_reject(
    message: str, *, symbol: str, requested: int, effective: int, lot: int,
    trace: str, signal: str,
) -> TradePlanSubmitResult:
    result = _reject(message)
    result.details = {
        **(result.details or {}),
        "entry_sizing_requested_quantity": requested,
        "entry_sizing_effective_quantity": effective,
        "entry_sizing_lot_size": lot,
        "entry_sizing_symbol": symbol,
        "entry_sizing_trace_id": trace,
        "entry_sizing_signal_id": signal,
    }
    return result


def _plan_for(symbol: str, quantity: int, trace: str, signal: str) -> TradePlan:
    return TradePlan(
        symbol=symbol,
        side="BUY",  # type: ignore[arg-type]
        quantity=quantity,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        signal_id=signal,
        trace_id=trace,
        tag="runner-entry",
    )


def test_recovery_sizing_records_do_not_cross_contaminate(monkeypatch) -> None:
    """Two trades on ONE manager must each freeze to their own record.

    Against e73e0461 the fallback read manager._last_entry_sizing_details --
    the most recent submission -- so trade A's retry could consume trade B's
    sizing. Provenance is now owned by each individual result.
    """
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    a_reject = _sized_reject(
        "margin required 20000 available 5000",
        symbol="NFO:NIFTYA", requested=130, effective=65, lot=65,
        trace="trace-A", signal="signal-A",
    )
    b_reject = _sized_reject(
        "margin required 30000 available 5000",
        symbol="NFO:NIFTYB", requested=200, effective=100, lot=50,
        trace="trace-B", signal="signal-B",
    )
    manager = _Manager(
        [a_reject, _accept("A-retry"), b_reject, _accept("B-retry")],
        resolver=_Resolver(lot=65),
        margin=10_000_000.0,
    )
    # A stale manager-global cache from some OTHER trade. Against e73e0461
    # _first_effective_quantity() fell back to this whenever the result's own
    # details lacked the key, letting one trade consume another's sizing.
    # It must never be consulted now.
    manager._last_entry_sizing_details = {
        "entry_sizing_requested_quantity": 200,
        "entry_sizing_effective_quantity": 100,
        "entry_sizing_lot_size": 50,
    }

    manager.submit_trade_plan_result(
        _plan_for("NFO:NIFTYA", 130, "trace-A", "signal-A")
    )
    a_retry = manager.plans[1]
    assert a_retry.quantity <= 65

    manager._instrument_resolver = _Resolver(lot=50)
    manager.submit_trade_plan_result(
        _plan_for("NFO:NIFTYB", 200, "trace-B", "signal-B")
    )
    b_retry = manager.plans[3]
    assert b_retry.quantity <= 100
    # B must not have been squeezed down to A's 65-unit record.
    assert b_retry.quantity > 65


def test_recovery_rejects_mismatched_sizing_provenance(monkeypatch) -> None:
    """A record belonging to another trade must be refused, with no retry."""
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    # Plan is A; the result carries B's sizing record.
    wrong = _sized_reject(
        "margin required 20000 available 5000",
        symbol="NFO:NIFTYB", requested=200, effective=100, lot=50,
        trace="trace-B", signal="signal-B",
    )
    manager = _Manager(
        [wrong, _accept("should-not-happen")],
        resolver=_Resolver(lot=65),
        margin=10_000_000.0,
    )

    result = manager.submit_trade_plan_result(
        _plan_for("NFO:NIFTYA", 130, "trace-A", "signal-A")
    )

    # Exactly one submission: the retry was refused on provenance grounds.
    assert len(manager.plans) == 1
    assert result.accepted is False
    recovery = (result.details or {}).get("entry_recovery") or {}
    assert recovery.get("outcome") == "entry_sizing_provenance_invalid"


def test_recovery_ignores_manager_global_sizing_cache(monkeypatch) -> None:
    """A legacy result with no sizing record must not borrow manager state.

    Against e73e0461 the manager-global cache supplied a cap here, so the
    retry was silently sized from an unrelated trade's record.
    """
    monkeypatch.delenv("DEFAULT_OPTION_LOT_SIZE", raising=False)
    legacy = _reject("margin required 20000 available 5000")  # no sizing details
    manager = _Manager(
        [legacy, _accept("retry")],
        resolver=_Resolver(lot=65),
        margin=10_000_000.0,
    )
    manager._last_entry_sizing_details = {
        "entry_sizing_requested_quantity": 130,
        "entry_sizing_effective_quantity": 65,
        "entry_sizing_lot_size": 65,
    }

    manager.submit_trade_plan_result(_plan_for("NFO:NIFTYA", 130, "t", "s"))

    retry = manager.plans[1]
    # Legacy behaviour (resolver lot 65, normal resize) -- NOT a cap borrowed
    # from the manager cache. The affordable resize exceeds 65 here.
    assert retry.quantity > 65
    assert retry.quantity % 65 == 0
