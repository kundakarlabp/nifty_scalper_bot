from __future__ import annotations

import time

from nifty_scalper_bot.core.strategy_manager import Signal, StrategyManager, StrategyVote
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


_SYMBOL = "NFO:NIFTY2670724050CE"


def _signal_vote(
    strategy: str,
    *,
    side: str = "CE",
    raw_score: float,
    weighted_score: float,
    confidence: float,
    role: str = "trigger",
) -> tuple[Signal, StrategyVote]:
    signal = Signal(
        action="BUY",
        symbol=_SYMBOL,
        quantity=65,
        confidence=confidence,
        reason=strategy,
        stop_loss=100.0,
        take_profit=130.0,
        metadata={
            "strategy": strategy,
            "is_selected_option": True,
            "quote_depth_valid": True,
            "tradable_quote": True,
            "spread_pct": 0.2,
        },
    )
    metadata = {
        "role": role,
        "raw_setup_score": raw_score,
        "raw_vote_score": raw_score,
        "regime_weight": weighted_score / raw_score,
        "regime_weighted_vote_score": weighted_score,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "spread_pct": 0.2,
    }
    if role == "trigger" and strategy == "VWAPPro":
        # Reproduce the 14:20 live-quality shape: 6.83 before independent
        # trigger confirmation (2.833 setup + 2 direction + 1 freshness +
        # 1 same-side context).
        metadata["direction_alignment_score"] = 2.0
    if role == "context":
        metadata.update(
            {
                "context_bonus_score": 2.0,
                "vote_timestamp": time.time(),
                "trigger_conditions_met": False,
                "trigger_block_reason": "context_only_role",
            }
        )
    vote = StrategyVote(
        strategy=strategy,
        side=side,
        score=weighted_score,
        confidence=confidence,
        reasons=[],
        metadata=metadata,
    )
    return signal, vote


def _live_indicators() -> dict[str, object]:
    return {
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "underlying_direction_confidence": 0.70,
        "context_fresh": True,
        "context_age_seconds": 0.1,
        "selected_ce": _SYMBOL,
        "is_selected_option": True,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "spread_pct": 0.2,
        "stale_data_used": False,
    }


def test_aligned_independent_trigger_can_clear_unchanged_live_quality_floor(monkeypatch) -> None:
    """A second independent trigger is bounded quality evidence, not a lower floor."""
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    manager = StrategyManager.__new__(StrategyManager)
    manager._last_no_signal_decision_by_symbol = {}

    vwap = _signal_vote(
        "VWAPPro", raw_score=8.5, weighted_score=6.8, confidence=0.85
    )
    orb = _signal_vote(
        "ORBPro", raw_score=7.0, weighted_score=5.6, confidence=0.70
    )
    orderflow = _signal_vote(
        "OrderFlow",
        raw_score=8.0,
        weighted_score=8.0,
        confidence=0.80,
        role="context",
    )

    result = manager._combine_strategy_votes(
        symbol=_SYMBOL,
        signals=[vwap, orb, orderflow],
        indicators=_live_indicators(),
    )

    assert result is not None
    assert result.metadata["quality_min_required"] == 7.0
    assert result.metadata["trade_quality_components"]["independent_trigger_confirmation"] == 0.5
    assert result.metadata["trade_quality_score"] >= 7.0
    assert result.metadata["approval_path"] == "aligned_two_trigger_consensus"


def test_opposite_trigger_does_not_receive_quality_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    manager = StrategyManager.__new__(StrategyManager)
    manager._last_no_signal_decision_by_symbol = {}

    vwap = _signal_vote(
        "VWAPPro", raw_score=8.5, weighted_score=6.8, confidence=0.85
    )
    opposite_orb = _signal_vote(
        "ORBPro", side="PE", raw_score=7.0, weighted_score=5.6, confidence=0.70
    )
    orderflow = _signal_vote(
        "OrderFlow",
        raw_score=8.0,
        weighted_score=8.0,
        confidence=0.80,
        role="context",
    )

    result = manager._combine_strategy_votes(
        symbol=_SYMBOL,
        signals=[vwap, opposite_orb, orderflow],
        indicators=_live_indicators(),
    )

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol[_SYMBOL]
    assert decision.blocked_at == "trade_quality_gate"


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
    """Optional context keeps full ticks/OHLC but its age alone cannot block entry."""
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
