from __future__ import annotations

from datetime import datetime, timezone
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
