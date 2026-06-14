from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner, SymbolState
from nifty_scalper_bot.strategies.signal_generator import Signal


def _runner() -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None, error=lambda *a, **k: None)
    r._data_hub = None
    r._market_data = None
    r._active_symbols = set()
    r._tracked_symbols = set()
    r._data_phase = {}
    r._symbol_states = {}
    r._last_bar_ts = {}
    r._universe_controller = SimpleNamespace(update=lambda *_: None)
    r._lock = SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s,a,b,c: None)
    r._running = False
    r._strategy_manager = SimpleNamespace(track_symbol=lambda *_: None)
    r._option_required_bars = 3
    r._context_required_bars = 3
    r._required_candles = 3
    r._indicator_engine = SimpleNamespace(get_history=lambda _s: [], get_indicators=lambda _s: {'rsi': 70.0, 'vwap': 100.0, 'ema': 99.0})
    r._set_symbol_hydration_state = lambda _s, st: st
    r.ingest_historical_bar = lambda _b: None
    r._has_session_candle_gaps = lambda _s: False
    r._symbol_bar_count = {}
    r._hydration_ready_streak = {}
    r._last_tick_time_by_symbol = {}
    r._premium_squeeze_last_signal_ts = {}
    r._reason_last_signal_ts = {}
    r._underlying_signal_cooldown_seconds = 30.0
    r._cooldown_log_throttle_seconds = 1.0
    r._vwap_sl_pct = 1.0
    r._vwap_tp_pct = 2.0
    return r


def test_hydrate_missing_bars_cache_only() -> None:
    r = _runner()
    bad = MagicMock(side_effect=AssertionError('fetch_history should not be called'))
    r._data_hub = SimpleNamespace(fetch_history=bad, get_ohlc_bars=lambda *_a, **_k: [{'timestamp': datetime.now(timezone.utc), 'open': 1, 'high': 2, 'low': 1, 'close': 2, 'volume': 10}])
    rows = r._hydrate_missing_bars('NSE:NIFTY', 1)
    assert rows
    bad.assert_not_called()


def test_add_symbol_uses_mdm_cache_not_fetch_history() -> None:
    r = _runner()
    r._normalize_symbol = lambda s: s
    r._symbol_state = {}
    r._candle_engines = {}
    r._frozen_universe = set()
    r._subscribe_symbol = lambda *_: None
    r._hydrate_from_mdm_cache = MagicMock(return_value=0)
    r._data_hub = SimpleNamespace(fetch_history=MagicMock(side_effect=AssertionError("no fetch_history")))
    r._market_data = SimpleNamespace(fetch_history=MagicMock(side_effect=AssertionError("no fetch_history")))
    r.add_symbol("NSE:NIFTY")
    r._hydrate_from_mdm_cache.assert_called_once()


def test_update_symbol_hydration_ready_without_live_tick() -> None:
    r = _runner()
    out = r.update_symbol_hydration('NSE:NIFTY', [1.0, 2.0, 3.0], {'NSE:NIFTY': {'vwap': 0.0, 'cum_volume': 0.0}})
    assert out == SymbolState.READY


def test_premium_squeeze_emission_sets_cooldown_and_metadata() -> None:
    r = _runner()
    sig = r._maybe_generate_premium_squeeze_signal('NFO:NIFTY26APR23800CE', 110.0, trace_id='t1')
    assert sig is not None
    assert sig.confidence == 0.72
    assert r._premium_squeeze_last_signal_ts.get('NIFTY', 0.0) > 0.0
    assert sig.metadata.get('strategy_name') == 'premium_momentum_squeeze'


def test_ltp_only_candidate_does_not_fake_spread() -> None:
    r = _runner()
    r._market_data = SimpleNamespace(get_symbol_snapshot=lambda _s: SimpleNamespace(ltp=100.0, bid=None, ask=None, tick_age_s=1.0, real_ticks_last_60s=1, tradable_quote=False, source='x'))
    sig = Signal(action='BUY', symbol='NFO:NIFTY26APR23800CE', quantity=1, confidence=0.8, reason='x', stop_loss=95.0, take_profit=110.0, metadata={})
    c = r._build_single_candidate_from_signal(signal=sig, metadata={}, option_side='CE')
    assert c is not None
    assert c['spread_pct'] is None
    assert c['bid'] == 0.0 and c['ask'] == 0.0


def test_get_spot_tick_alias_and_snapshot_fallback() -> None:
    r = _runner()
    r._market_data = SimpleNamespace(
        get_latest_tick=lambda s: {"symbol": s, "ltp": 1} if s == "NIFTY50" else None,
        get_symbol_snapshot=lambda _s: {"ltp": 25000, "received_at": 10.0},
        get_cached_ltp=lambda *_a, **_k: None,
    )
    r._data_hub = SimpleNamespace(get_latest_tick=lambda *_: None, get_quote=lambda *_: None)
    assert r._get_spot_tick()["symbol"] == "NIFTY50"
    r._market_data = SimpleNamespace(
        get_latest_tick=lambda *_: None,
        get_symbol_snapshot=lambda _s: {"ltp": 25010, "received_at": 20.0},
        get_cached_ltp=lambda *_a, **_k: None,
    )
    tick = r._get_spot_tick()
    assert tick and tick["symbol"] == "NSE:NIFTY" and tick["received_at"] == 20.0


def test_update_symbol_hydration_soft_data_uses_runner_logger(monkeypatch) -> None:
    r = _runner()
    r._has_session_candle_gaps = lambda _s: True
    r._session_gap_count = {'NSE:NIFTY': 2}
    r._last_tick_time_by_symbol = {'NSE:NIFTY': 0.0}
    captured = {}

    def _capture(logger, *_args, **_kwargs):
        captured['logger'] = logger

    monkeypatch.setattr('nifty_scalper_bot.strategies.runner.log_throttled', _capture)
    r.update_symbol_hydration('NSE:NIFTY', [1.0, 2.0, 3.0], {'NSE:NIFTY': {'vwap': 1.0, 'cum_volume': 1.0}})
    assert captured['logger'] is r._logger


def test_get_spot_tick_snapshot_object_fallback() -> None:
    r = _runner()
    snap = SimpleNamespace(ltp=25010.0, tick_age_s=1.5, source='snapshot_obj')
    r._market_data = SimpleNamespace(get_latest_tick=lambda *_: None, get_symbol_snapshot=lambda _s: snap, get_cached_ltp=lambda *_a, **_k: None)
    r._data_hub = SimpleNamespace(get_latest_tick=lambda *_: None, get_quote=lambda *_: None)
    tick = r._get_spot_tick()
    assert tick is not None
    assert tick['symbol'] == 'NSE:NIFTY'
    assert tick['ltp'] == 25010.0
    assert tick['source'] == 'snapshot_obj'


def test_repair_candle_gaps_requests_mdm_hydration() -> None:
    r = _runner()
    r._main_loop = None
    r._gap_repair_inflight = set()
    r._required_bars_for_symbol = lambda _s: 33
    called = []
    r._request_mdm_hydration = lambda symbol, target: called.append((symbol, target))
    prev = SimpleNamespace(start=datetime(2026,1,1,9,15,tzinfo=timezone.utc), end=datetime(2026,1,1,9,15,59,tzinfo=timezone.utc), close=100.0)
    curr = SimpleNamespace(start=datetime(2026,1,1,9,20,tzinfo=timezone.utc), end=datetime(2026,1,1,9,20,59,tzinfo=timezone.utc), close=101.0)
    r._repair_candle_gap('NSE:NIFTY', prev, curr)
    assert called == [('NSE:NIFTY', 33)]


def test_refresh_history_if_due_requests_mdm_hydration() -> None:
    r = _runner()
    loop = asyncio.new_event_loop()
    try:
        r._main_loop = loop
        r._last_history_refresh_by_symbol = {}
        r._history_refresh_interval_seconds = 0.0
        r._required_bars_for_symbol = lambda _s: 21
        called = []
        r._request_mdm_hydration = lambda symbol, target: called.append((symbol, target))
        r._refresh_history_if_due('NSE:NIFTY')
        assert called == [('NSE:NIFTY', 21)]
    finally:
        loop.close()


def test_runner_emergency_backfill_disabled_by_default(monkeypatch) -> None:
    r = _runner()
    r._config = SimpleNamespace(fetch_history_on_startup=True)
    r._backfill_task_started = False
    r._main_loop = asyncio.new_event_loop()
    r._subscribe_symbol = lambda *_: None
    r._symbols = []
    r._callbacks = {}
    r._market_data = SimpleNamespace(start=lambda: None)
    r._lock = type('L', (), {'__enter__': lambda s: None, '__exit__': lambda s,a,b,c: None})()
    monkeypatch.delenv('RUNNER_ENABLE_EMERGENCY_BACKFILL', raising=False)
    try:
        r.start([])
        assert r._backfill_task_started is False
    finally:
        r._main_loop.close()

from datetime import timedelta

from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def _bars(count: int, *, start_close: float = 100.0) -> list[dict]:
    base = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    return [
        {
            "timestamp": base + timedelta(minutes=i),
            "open": start_close + i,
            "high": start_close + i + 1,
            "low": start_close + i - 1,
            "close": start_close + i + 0.5,
            "volume": 100 + i,
        }
        for i in range(count)
    ]


def test_datahub_context_bars_sync_into_indicator_engine() -> None:
    r = _runner()
    r._indicator_engine = IndicatorEngine()
    r._market_data = None
    r._data_hub = SimpleNamespace(
        get_ohlc_bars=lambda symbol, limit=None: _bars(50, start_close=200.0 if symbol == "NFO:NIFTY26JUNFUT" else 100.0)[-(limit or 50):]
    )
    r._symbol_history = {}
    r._context_required_bars = 50
    r._option_required_bars = 5
    r._active_symbols = {"NSE:NIFTY", "NFO:NIFTY26JUNFUT"}
    r._active_futures_symbol = "NFO:NIFTY26JUNFUT"
    r._tracked_symbols = set()
    r._data_phase = {}
    r._last_bar_ts = {}
    r._symbol_states = {}
    r._hydration_attempted_symbols = set()
    r._last_hydration_reason_by_symbol = {}
    r._set_symbol_hydration_state = lambda _s, st: st
    r._seed_pipeline_store = lambda _s: None
    r._seed_candle_engine_from_history = lambda _s: None

    r._sync_context_history_if_cold(source="test_context_sync")

    assert len(r._indicator_engine.get_history("NSE:NIFTY")) >= 50
    assert len(r._indicator_engine.get_history("NFO:NIFTY26JUNFUT")) >= 50


def test_selected_option_prewarm_schedules_canonical_runtime_ensure() -> None:
    r = _runner()
    r._indicator_engine = IndicatorEngine()
    r._selected_option_prewarm_last = {}
    r._selected_option_prewarm_inflight = set()
    r._selected_option_prewarm_cooldown_s = 0.0
    symbol = "NFO:NIFTY26JUN24000CE"
    calls: list[dict] = []
    r._runtime_history_ensurer = lambda sym, **kwargs: calls.append({"symbol": sym, **kwargs})
    r._data_hub = SimpleNamespace()

    r._request_selected_option_history_prewarm(symbol, bars_before=0, required_bars=5)

    assert calls and calls[-1]["symbol"] == symbol
    assert calls[-1]["target_bars"] == 5


def test_runner_sync_history_from_mdm_is_canonical_transition() -> None:
    r = _runner()
    r._normalize_symbol = lambda s: s
    r._symbol_history = {}
    r._indicator_engine = IndicatorEngine()
    r._set_symbol_hydration_state = lambda _s, st: st
    r._seed_pipeline_store = lambda _s: None
    r._seed_candle_engine_from_history = lambda _s: None
    r._get_mdm_bars = lambda _symbol, _limit: _bars(5)
    result = r.sync_history_from_mdm('NSE:NIFTY', required_bars=5, reason='test', role='spot_context')
    assert result.success is True
    assert result.mdm_bars == 5
    assert result.runner_bars >= 5
    assert result.indicator_bars >= 5
