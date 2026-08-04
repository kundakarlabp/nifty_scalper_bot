from __future__ import annotations

import logging

from nifty_scalper_bot.core.boot_log_safety import BootLogRateControl


def _record(
    event: str,
    *,
    logger_name: str = "nifty_scalper_bot.strategies.elite_strategies.order_flow",
    message: str | None = None,
    **extra: object,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message or event,
        args=(),
        exc_info=None,
    )
    record.event = event
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_alternating_option_symbols_are_throttled_independently() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    ce = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )
    pe = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000PE",
        side="PE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )

    assert control.filter(ce) is True
    assert control.filter(pe) is True
    assert control.filter(ce) is False
    assert control.filter(pe) is False


def test_option_state_change_is_visible_after_other_symbol() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    ce_blocked = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )
    pe_blocked = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000PE",
        side="PE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )
    ce_changed = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=True,
        trigger_block_reason="",
    )

    assert control.filter(ce_blocked) is True
    assert control.filter(pe_blocked) is True
    assert control.filter(ce_changed) is True


def test_orderflow_elite_result_is_relabelled_as_context_vote() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    record = _record(
        "elite_strategy_signal",
        logger_name="nifty_scalper_bot.strategies.elite_strategies.base_elite",
        message="Condition met: elite signal generated",
        strategy="OrderFlow",
    )

    assert control.filter(record) is True
    assert record.event == "elite_strategy_context_vote"
    assert record.getMessage() == "Condition met: elite context vote generated"
    assert record.role == "context"
    assert record.can_trigger is False
    assert record.context_only is True


def test_trigger_strategy_signal_label_is_preserved() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    record = _record(
        "elite_strategy_signal",
        logger_name="nifty_scalper_bot.strategies.elite_strategies.base_elite",
        message="Condition met: elite signal generated",
        strategy="VWAPPro",
    )

    assert control.filter(record) is True
    assert record.event == "elite_strategy_signal"
    assert record.getMessage() == "Condition met: elite signal generated"
    assert not hasattr(record, "context_only")


def test_orderflow_decision_explicitly_publishes_context_only_role() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    record = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=True,
        trigger_block_reason="",
    )

    assert control.filter(record) is True
    assert record.role == "context"
    assert record.can_trigger is False
    assert record.context_only is True
    assert record.contract_side == "CE"


def test_global_readiness_throttle_remains_global() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "LIVE_READINESS_COMPUTED",
        logger_name="nifty_scalper_bot.execution.readiness",
        ready=False,
        primary_blocker="history_cold",
    )
    same = _record(
        "LIVE_READINESS_COMPUTED",
        logger_name="nifty_scalper_bot.execution.readiness",
        ready=False,
        primary_blocker="history_cold",
    )
    changed = _record(
        "LIVE_READINESS_COMPUTED",
        logger_name="nifty_scalper_bot.execution.readiness",
        ready=True,
        primary_blocker=None,
    )

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(changed) is True
