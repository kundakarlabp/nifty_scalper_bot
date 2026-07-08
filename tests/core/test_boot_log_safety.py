from __future__ import annotations

import logging

from nifty_scalper_bot.core.boot_log_safety import BootLogRateControl
from nifty_scalper_bot.core.boot_readiness_safety import adapt_compute_live_readiness
from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


def _record(event: str, **extra):
    record = logging.LogRecord(
        name="nifty_scalper_bot.core.instrument_manager",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.event = event
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_rate_control_allows_changed_basket_state() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record("CONTRACT_SSOT_BASKET_SELECTED", selected_ce="CE1", selected_pe="PE1", atm_strike=24250)
    same = _record("CONTRACT_SSOT_BASKET_SELECTED", selected_ce="CE1", selected_pe="PE1", atm_strike=24250)
    changed = _record("CONTRACT_SSOT_BASKET_SELECTED", selected_ce="CE2", selected_pe="PE2", atm_strike=24300)

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(changed) is True


def test_rate_control_covers_bootstrap_state() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record("LIVE_UNIVERSE_BOOTSTRAP_STATUS", symbol="NSE:NIFTY", ready=False, reason="waiting")
    same = _record("LIVE_UNIVERSE_BOOTSTRAP_STATUS", symbol="NSE:NIFTY", ready=False, reason="waiting")

    assert control.filter(first) is True
    assert control.filter(same) is False


def test_rate_control_covers_orderflow_direction_context_noise() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )
    same = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )
    changed = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=True,
        trigger_block_reason="",
    )

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(changed) is True


def test_rate_control_covers_live_validation_checklist() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "LIVE_VALIDATION_CHECKLIST",
        primary_blocker="selected_option_quote_missing",
        blockers=("selected_option_quote_missing",),
        evaluation_ready=True,
        execution_ready=False,
        live_orders_armed=False,
    )
    same = _record(
        "LIVE_VALIDATION_CHECKLIST",
        primary_blocker="selected_option_quote_missing",
        blockers=("selected_option_quote_missing",),
        evaluation_ready=True,
        execution_ready=False,
        live_orders_armed=False,
    )
    changed = _record(
        "LIVE_VALIDATION_CHECKLIST",
        primary_blocker=None,
        blockers=(),
        evaluation_ready=True,
        execution_ready=True,
        live_orders_armed=True,
    )

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(changed) is True


def test_live_validation_checklist_log_contains_primary_gate(caplog) -> None:
    caplog.set_level(logging.INFO, logger="nifty_scalper_bot.execution.readiness")

    decision = normalize_readiness_blockers(
        ["selected_ce_quote_missing"],
        "OPEN",
        broker_state={"broker_balance_valid": True},
        live_mode=True,
        evaluation_ready=True,
        execution_ready=False,
    )

    record = next(r for r in caplog.records if getattr(r, "event", "") == "LIVE_VALIDATION_CHECKLIST")
    assert decision.primary_blocker == "selected_option_quote_missing"
    assert record.primary_blocker == "selected_option_quote_missing"
    assert record.evaluation_ready is True
    assert record.execution_ready is False
    assert record.live_orders_armed is False


def test_session_readiness_adapter_removes_option_details_outside_session() -> None:
    def original(**kwargs):
        reasons = []
        if not kwargs["market_open"]:
            reasons.append("market_closed")
        if not kwargs["ce_quote_ready"]:
            reasons.append("ce_quote")
        if not kwargs["pe_quote_ready"]:
            reasons.append("pe_quote")
        if kwargs["ce_bars"] < kwargs["option_exec_min_bars"]:
            reasons.append("ce_history")
        if kwargs["pe_bars"] < kwargs["option_exec_min_bars"]:
            reasons.append("pe_history")
        return False, reasons

    adapted = adapt_compute_live_readiness(original)
    _armed, reasons = adapted(
        live_mode=True,
        market_open=False,
        ce_quote_ready=False,
        pe_quote_ready=False,
        ce_bars=0,
        pe_bars=0,
        option_exec_min_bars=30,
    )

    assert reasons == ["market_closed"]
