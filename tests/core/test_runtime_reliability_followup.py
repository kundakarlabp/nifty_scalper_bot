from __future__ import annotations

import time
import types
from unittest.mock import MagicMock

from nifty_scalper_bot.core.app import _reconciliation_sleep_seconds
from nifty_scalper_bot.core.strategy_manager import Signal, StrategyManager, StrategyVote
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _wire_mdm() -> tuple[MarketDataManager, str, str]:
    mdm = MarketDataManager(kite=None)
    selected = "NFO:NIFTY26AUG24550CE"
    far = "NFO:NIFTY26AUG25000CE"
    mapping = {
        1: selected,
        2: "NFO:NIFTY26AUG24550PE",
        3: "NSE:NIFTY",
        4: "NFO:NIFTY26AUGFUT",
        5: far,
    }
    for token, symbol in mapping.items():
        mdm._symbol_by_token[token] = symbol
        mdm._token_to_symbol[token] = symbol
        mdm._symbol_to_token[symbol] = token
        mdm._token_by_symbol[symbol] = token
    mdm.set_active_contract_basket(
        {
            "all_tokens": [1, 2, 3, 4],
            "token_by_symbol": {
                selected: 1,
                "NFO:NIFTY26AUG24550PE": 2,
                "NSE:NIFTY": 3,
                "NFO:NIFTY26AUGFUT": 4,
            },
            "spot_symbol": "NSE:NIFTY",
            "spot_token": 3,
            "futures_symbol": "NFO:NIFTY26AUGFUT",
            "selected_ce": selected,
            "selected_pe": "NFO:NIFTY26AUG24550PE",
            "option_symbols": [selected, "NFO:NIFTY26AUG24550PE"],
        }
    )
    mdm._overload_enter_oldest_ms = 100.0
    mdm._overload_exit_oldest_ms = 40.0
    return mdm, selected, far


def _pending_tick(symbol: str, age_s: float = 1.0) -> dict[str, object]:
    return {
        "symbol": symbol,
        "last_price": 100.0,
        "_mdm_enqueued_mono": time.monotonic() - age_s,
    }


def test_stale_far_context_tick_does_not_disarm_healthy_entry_pipeline() -> None:
    mdm, _selected, far = _wire_mdm()
    tick = _pending_tick(far)
    with mdm._pending_tick_lock:
        mdm._pending_far_ticks[far] = tick
        mdm._pending_tick_count = 1
        mdm._pending_heap_push_locked(tick, far)
        mdm._update_pipeline_overload_locked()

    assert mdm.pipeline_overloaded is False


def test_stale_selected_option_tick_still_trips_overload_fail_closed() -> None:
    mdm, selected, _far = _wire_mdm()
    tick = _pending_tick(selected)
    with mdm._pending_tick_lock:
        mdm._pending_tick_queues[selected].append(tick)
        mdm._pending_tick_count = 1
        mdm._pending_heap_push_locked(tick, selected)
        mdm._update_pipeline_overload_locked()

    assert mdm.pipeline_overloaded is True


def test_large_far_context_backlog_still_trips_global_count_fail_closed() -> None:
    mdm, _selected, far = _wire_mdm()
    mdm._overload_enter_pending = 1
    tick = _pending_tick(far, age_s=0.01)
    with mdm._pending_tick_lock:
        mdm._pending_far_ticks[far] = tick
        mdm._pending_tick_count = 1
        mdm._pending_heap_push_locked(tick, far)
        mdm._update_pipeline_overload_locked()

    assert mdm.pipeline_overloaded is True


def test_reconciliation_freshness_cap_remains_fail_closed(monkeypatch) -> None:
    """A 30 s max-age intentionally requires a <=15 s refresh while market is open."""
    monkeypatch.setenv("HEALTH_FLAT_RECONCILE_INTERVAL_SEC", "60")
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "30")
    ctx = types.SimpleNamespace(
        position_reconciliation_failed=False,
        unresolved_reconciliation_symbols=set(),
        unprotected_broker_positions=set(),
        unprotected_broker_position=False,
        position_manager=types.SimpleNamespace(
            get_open_positions=lambda: [], get_pending_orders=lambda: []
        ),
    )

    assert _reconciliation_sleep_seconds(ctx, market_open=True) == 15.0


def _manager_probe() -> StrategyManager:
    manager = StrategyManager.__new__(StrategyManager)
    manager._last_no_signal_decision_by_symbol = {}
    manager._compute_trade_quality_score = lambda *args, **kwargs: (10.0, {})
    return manager


def _downweighted_trigger() -> tuple[Signal, StrategyVote]:
    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY26AUG24550CE",
        quantity=65,
        confidence=0.9,
        reason="ORBPro",
        stop_loss=140.0,
        take_profit=150.0,
        metadata={
            "strategy": "ORBPro",
            "is_selected_option": True,
            "quote_depth_valid": True,
            "spread_pct": 0.2,
        },
    )
    vote = StrategyVote(
        strategy="ORBPro",
        side="CE",
        score=5.6,
        confidence=0.8,
        reasons=[],
        metadata={
            "role": "trigger",
            "raw_setup_score": 8.0,
            "raw_vote_score": 8.0,
            "regime_weight": 0.7,
            "regime_weighted_vote_score": 5.6,
        },
    )
    return signal, vote


def test_weighted_score_rejection_is_not_mislabeled_as_raw_score_failure(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "true")
    manager = _manager_probe()

    result = manager._combine_strategy_votes(
        symbol="NFO:NIFTY26AUG24550CE",
        signals=[_downweighted_trigger()],
        indicators={
            "direction_bias": "CE",
            "underlying_direction_bias": "CE",
            "context_fresh": True,
            "context_age_seconds": 0.1,
            "selected_ce": "NFO:NIFTY26AUG24550CE",
            "is_selected_option": True,
            "quote_depth_valid": True,
            "spread_pct": 0.2,
        },
    )

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol["NFO:NIFTY26AUG24550CE"]
    assert decision.reason == "regime_weighted_score_below_min"


def test_cpu_summary_counts_dynamic_active_options_when_whitelist_is_empty() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._cpu_opt_metrics = {}
    runner._eval_option_whitelist = set()
    runner._active_symbols = {
        "NSE:NIFTY",
        "NFO:NIFTY26AUGFUT",
        "NFO:NIFTY26AUG24550CE",
        "NFO:NIFTY26AUG24550PE",
    }
    runner._logger = MagicMock()
    runner._should_log_throttled = lambda *_args, **_kwargs: True

    runner._bump_cpu_metric("evaluated_symbols")

    _args, kwargs = runner._logger.info.call_args
    assert kwargs["extra"]["active_option_symbols_count"] == 2


def test_datahub_mdm_canonical_tick_skips_expensive_recanonicalization(monkeypatch) -> None:
    """MDM's real normalized WS shape must not be rebuilt inside DataHub."""
    mdm = types.SimpleNamespace(attach_tick_bus=lambda _bus: None)
    hub = DataHub(mdm)
    symbol = "NFO:NIFTY26AUG24550CE"
    now = time.time()
    tick = {
        "symbol": symbol,
        "instrument_token": 101,
        "token": 101,
        "ltp": 123.45,
        "last_price": 123.45,
        "timestamp": "2026-08-07T09:45:00+00:00",
        "timestamp_ms": 1786095900000.0,
        "timestamp_source": "exchange_timestamp",
        "source_timestamp_valid": True,
        "received_at": now,
        "source": "ws_full",
        "bid": 123.40,
        "ask": 123.50,
        "best_bid": 123.40,
        "best_ask": 123.50,
        "spread": 0.10,
        "depth": {"buy": [{"price": 123.40}], "sell": [{"price": 123.50}]},
        "depth_available": True,
        "tradable_quote": True,
        "bid_missing": False,
        "ask_missing": False,
        "bid_ask_source": "ws_full",
    }

    def _should_not_restamp(*_args, **_kwargs):
        raise AssertionError("canonical MDM tick was redundantly restamped")

    monkeypatch.setattr(hub, "_stamp_quote_identity", _should_not_restamp)
    hub.ingest_tick_sync(tick)

    assert hub._quotes[symbol]["ltp"] == 123.45
    assert hub._quotes[symbol]["timestamp_ms"] == 1786095900000.0
    assert hub._quotes[symbol]["source_timestamp_valid"] is True


def test_datahub_generic_tick_still_uses_full_canonicalization(monkeypatch) -> None:
    """The optimisation must not weaken validation for non-MDM/raw tick callers."""
    mdm = types.SimpleNamespace(attach_tick_bus=lambda _bus: None)
    hub = DataHub(mdm)
    original = hub._stamp_quote_identity
    stamp_calls = 0

    def _stamp(*args, **kwargs):
        nonlocal stamp_calls
        stamp_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hub, "_stamp_quote_identity", _stamp)
    hub.ingest_tick(
        {
            "symbol": "NFO:NIFTY26AUG24550PE",
            "instrument_token": 102,
            "last_price": 110.0,
            "timestamp": "2026-08-07T09:45:00+00:00",
            "source": "ws",
        }
    )

    assert stamp_calls >= 1
