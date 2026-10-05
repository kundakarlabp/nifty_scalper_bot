from __future__ import annotations

import time

from nifty_scalper_bot.core.strategy_manager import (
    Signal,
    StrategyEvidence,
    StrategyManager,
)
from nifty_scalper_bot.data.market_data_manager import MarketDataManager

_CE = "NFO:NIFTY2670724050CE"
_PE = "NFO:NIFTY2670724050PE"


def _signal_evidence(
    strategy: str,
    *,
    symbol: str = _CE,
    side: str = "CE",
    role: str = "trigger",
    aligned_context: bool = True,
) -> tuple[Signal, StrategyEvidence]:
    signal = Signal(
        action="BUY",
        symbol=symbol,
        quantity=65,
        confidence=1.0,
        reason=strategy,
        stop_loss=100.0,
        take_profit=130.0,
        metadata={
            "strategy": strategy,
            "strategy_name": strategy,
            "role": role,
            "side": side,
            "trade_side": side,
            "contract_side": side,
            "setup_pass": role == "trigger",
            "trigger_conditions_met": role == "trigger",
            "is_selected_option": True,
            "quote_depth_valid": True,
            "tradable_quote": True,
            "spread_pct": 0.2,
            "required_data_present": True,
            "stale_data_used": False,
        },
    )
    metadata = dict(signal.metadata)
    if role == "context":
        metadata.update(
            {
                "setup_pass": True,
                "trigger_conditions_met": False,
                "trigger_block_reason": "context_only_role",
                "context_quality_eligible": True,
                "effective_context_alignment": aligned_context,
                "effective_context_conflict": not aligned_context,
                "vote_timestamp": time.time(),
            }
        )
    evidence = StrategyEvidence(
        strategy=strategy,
        side=side,
        reasons=["fixture"],
        metadata=metadata,
    )
    return signal, evidence


def _live_indicators(
    direction: str = "CE", *, transition: bool = False
) -> dict[str, object]:
    return {
        "direction_bias": direction,
        "underlying_direction_bias": direction,
        "underlying_direction_state": (
            "TRANSITION"
            if transition
            else ("CONFIRMED_BULL" if direction == "CE" else "CONFIRMED_BEAR")
        ),
        "context_fresh": True,
        "context_age_seconds": 0.1,
        "direction_context_source": "spot_futures_agree",
        "selected_ce": _CE,
        "selected_pe": _PE,
        "is_selected_option": True,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "spread_pct": 0.2,
        "stale_data_used": False,
    }


def _manager() -> StrategyManager:
    manager = StrategyManager.__new__(StrategyManager)
    manager._last_no_signal_decision_by_symbol = {}
    return manager


def test_countertrend_pe_is_blocked_during_confirmed_bull(monkeypatch) -> None:
    """Regression for the 2026-10-05 losing PE-in-uptrend failure mode."""
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    manager = _manager()
    pe_trigger = _signal_evidence("VWAPPro", symbol=_PE, side="PE")

    result = manager._combine_strategy_votes(
        symbol=_PE,
        signals=[pe_trigger],
        indicators=_live_indicators("CE"),
    )

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol[_PE]
    assert decision.reason == "countertrend_requires_structural_reversal_contract"
    assert decision.blocked_at == "direction_contract"


def test_direction_transition_blocks_new_entry(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    manager = _manager()
    ce_trigger = _signal_evidence("VWAPPro")

    result = manager._combine_strategy_votes(
        symbol=_CE,
        signals=[ce_trigger],
        indicators=_live_indicators("CE", transition=True),
    )

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol[_CE]
    assert decision.reason == "underlying_direction_transition"


def test_single_trigger_requires_fresh_independent_context(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    manager = _manager()
    trigger = _signal_evidence("VWAPPro")
    context = _signal_evidence("OrderFlow", role="context")

    result = manager._combine_strategy_votes(
        symbol=_CE,
        signals=[trigger, context],
        indicators=_live_indicators("CE"),
    )

    assert result is not None
    assert result.metadata["approval_path"] == "single_trigger_context_confirmed"
    assert result.metadata["direction_contract"]["passed"] is True
    assert result.metadata["setup_contract"]["passed"] is True
    assert result.metadata["confirmation_contract"]["passed"] is True
    assert result.metadata["context_confirmation_strategies"] == ["OrderFlow"]


def test_two_independent_same_side_triggers_can_confirm(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    manager = _manager()
    vwap = _signal_evidence("VWAPPro")
    momentum = _signal_evidence("premium_momentum_squeeze")

    result = manager._combine_strategy_votes(
        symbol=_CE,
        signals=[vwap, momentum],
        indicators=_live_indicators("CE"),
    )

    assert result is not None
    assert result.metadata["approval_path"] == "aligned_trigger_consensus"
    assert result.metadata["confirmation_contract"]["trigger_consensus"] is True
    assert result.metadata["confirming_trigger_strategies"] == [
        "premium_momentum_squeeze"
    ]


def test_opposite_trigger_sides_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    manager = _manager()
    ce = _signal_evidence("VWAPPro")
    pe = _signal_evidence("SMC", symbol=_PE, side="PE")

    result = manager._combine_strategy_votes(
        symbol=_CE,
        signals=[ce, pe],
        indicators=_live_indicators("CE"),
    )

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol[_CE]
    assert decision.reason == "trigger_side_conflict"


def _wired_mdm() -> tuple[MarketDataManager, str, str]:
    mdm = MarketDataManager(kite=None)
    selected = "NFO:NIFTY26AUG24550CE"
    selected_pe = "NFO:NIFTY26AUG24550PE"
    near_context = "NFO:NIFTY26AUG24600CE"
    mapping = {
        1: selected,
        2: selected_pe,
        3: "NSE:NIFTY",
        4: "NFO:NIFTY26AUGFUT",
        5: near_context,
    }
    for token, symbol in mapping.items():
        mdm._symbol_by_token[token] = symbol
        mdm._token_to_symbol[token] = symbol
        mdm._symbol_to_token[symbol] = token
        mdm._token_by_symbol[symbol] = token
    mdm.set_active_contract_basket(
        {
            "all_tokens": list(mapping),
            "token_by_symbol": {symbol: token for token, symbol in mapping.items()},
            "spot_symbol": "NSE:NIFTY",
            "futures_symbol": "NFO:NIFTY26AUGFUT",
            "selected_ce": selected,
            "selected_pe": selected_pe,
            "option_symbols": [selected, selected_pe, near_context],
        }
    )
    mdm._overload_enter_oldest_ms = 100.0
    mdm._overload_exit_oldest_ms = 40.0
    return mdm, selected, near_context


def _stale_pending(symbol: str, bucket: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "last_price": 100.0,
        "_mdm_priority_bucket": bucket,
        "_mdm_enqueued_mono": time.monotonic() - 1.0,
    }


def test_stale_nonselected_near_atm_context_does_not_disarm_entry() -> None:
    mdm, _selected, near_context = _wired_mdm()
    tick = _stale_pending(near_context, "near_atm")
    with mdm._pending_tick_lock:
        mdm._pending_tick_queues[near_context].append(tick)
        mdm._pending_tick_count = 1
        mdm._pending_heap_push_locked(tick, near_context)
        mdm._update_pipeline_overload_locked()
    assert mdm.pipeline_overloaded is False


def test_stale_selected_option_remains_age_critical() -> None:
    mdm, selected, _near_context = _wired_mdm()
    tick = _stale_pending(selected, "selected_option")
    with mdm._pending_tick_lock:
        mdm._pending_tick_queues[selected].append(tick)
        mdm._pending_tick_count = 1
        mdm._pending_heap_push_locked(tick, selected)
        mdm._update_pipeline_overload_locked()
    assert mdm.pipeline_overloaded is True


def test_unknown_normal_queue_remains_fail_closed() -> None:
    mdm, _selected, _near_context = _wired_mdm()
    unknown = "NFO:NIFTY26AUG24700CE"
    tick = {
        "symbol": unknown,
        "last_price": 100.0,
        "_mdm_enqueued_mono": time.monotonic() - 1.0,
    }
    with mdm._pending_tick_lock:
        mdm._pending_tick_queues[unknown].append(tick)
        mdm._pending_tick_count = 1
        mdm._pending_heap_push_locked(tick, unknown)
        mdm._update_pipeline_overload_locked()
    assert mdm.pipeline_overloaded is True
