from __future__ import annotations
import typing as t
from nifty_scalper_bot.core.strategy_manager import StrategyManager

class _S:
    name='VWAPPro'
    def __init__(self): self.last={}
    def get_required_indicators(self): return []
    def generate_signal(self, symbol:str, indicators:t.Mapping[str,t.Any], current_price:float, position:t.Any):
        self.last=dict(indicators)
        return None

class _IE:
    def __init__(self): self.payload={}
    def get_history(self, symbol:str): return [1]*30
    def get_indicators(self, symbol:str, names:t.Iterable[str]): return dict(self.payload)
class _PM:
    def get_position(self, symbol:str): return None


def test_context_propagates_direction_and_futures_fields():
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'vwap':100,'exchange_vwap':100,'volume':0,'avg_volume':0,'direction_bias':'CE'}
    m.generate_signal('NSE:NIFTY',100)
    ie.payload={'vwap':101,'exchange_vwap':101,'volume':100,'avg_volume':100,'futures_volume_ratio':1.7,'vwap_slope':0.1}
    m.generate_signal('NFO:NIFTY26MAYFUT',100)
    ie.payload={'vwap':102,'exchange_vwap':102,'volume':100,'avg_volume':100}
    m.generate_signal('NFO:NIFTY26MAY23500CE',100)
    assert st.last.get('direction_bias')=='CE'
    assert 'spot_context' in st.last
    assert st.last.get('futures_volume_ratio')==1.7
    assert st.last.get('futures_vwap')==101


def test_option_evaluation_receives_derived_spot_direction_bias():
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'close':110,'vwap':100,'ema_fast':105,'ema_slow':100,'ema_50':95,'vwap_slope':1.0,'volume':100,'avg_volume':100}
    m.generate_signal('NSE:NIFTY',110)
    ie.payload={'vwap':102,'exchange_vwap':102,'volume':100,'avg_volume':100}
    m.generate_signal('NFO:NIFTY26MAY23750CE',102)
    assert st.last.get('direction_bias')=='CE'
    assert st.last.get('underlying_direction_bias')=='CE'
    assert float(st.last.get('underlying_direction_confidence') or 0)>0
    assert float(st.last.get('context_age_seconds') or 999)<=120
    assert isinstance(st.last.get('spot_context'), dict)


def test_context_direction_fallback_from_previous_close():
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'close':110,'previous_close':100,'volume':100,'avg_volume':100}
    m.generate_signal('NSE:NIFTY',110)
    snap = m._latest_context_snapshots.get('spot_context', {})
    assert snap.get('direction_bias') == 'CE'
    assert float(snap.get('underlying_direction_confidence') or 0) >= 0.5
    assert float(snap.get('underlying_direction_confidence') or 0) <= 0.70


def test_spot_context_updates_without_vwap():
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'close':110,'previous_close':100,'volume':100,'avg_volume':100}
    m.generate_signal('NSE:NIFTY',110)
    snap = m._latest_context_snapshots.get('spot_context', {})
    assert snap.get('direction_bias') == 'CE'


def test_futures_context_updates_without_vwap_from_tick_slope():
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'close':100,'tick_slope':0.2,'volume':100,'avg_volume':100}
    m.generate_signal('NFO:NIFTY26MAYFUT',100)
    snap = m._latest_context_snapshots.get('futures_context', {})
    assert snap.get('direction_bias') == 'CE'


def test_option_context_fresh_but_directionless_not_used(caplog):
    import logging
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'close':100,'volume':100,'avg_volume':100}
    m.generate_signal('NSE:NIFTY',100)
    ie.payload={'vwap':102,'volume':100,'avg_volume':100}
    with caplog.at_level(logging.INFO):
        m.generate_signal('NFO:NIFTY26MAY23750CE',102)
    assert any("OPTION_CONTEXT_FRESH_BUT_DIRECTIONLESS" in rec.message for rec in caplog.records)
    assert any("direction_tie" in rec.message for rec in caplog.records)
    assert st.last.get('direction_bias') in (None, '')


def test_context_vote_logs_context_trigger_details(caplog):
    import logging
    from nifty_scalper_bot.core.strategy_manager import StrategyVote
    from nifty_scalper_bot.strategies.signal_generator import Signal
    m = StrategyManager([], _IE(), _PM())
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY23750PE', quantity=1, confidence=0.8, metadata={'role': 'context', 'trigger_block_reason': 'score_below_live_trigger_min'})
    vote = StrategyVote(strategy='OrderFlow', side='PE', confidence=0.8, score=8.0, metadata={'role': 'context', 'trigger_block_reason': 'score_below_live_trigger_min'})
    with caplog.at_level(logging.INFO):
        out = m._combine_strategy_votes(symbol='NFO:NIFTY26MAY23750PE', signals=[(signal, vote)], indicators={})
    assert out is None
    assert "context_trigger_details" in caplog.text
    assert "score_below_live_trigger_min" in caplog.text


def test_option_underlying_context_missing_log(caplog):
    import logging
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'vwap':102,'volume':100,'avg_volume':100}
    with caplog.at_level(logging.INFO):
        m.generate_signal('NFO:NIFTY26MAY23750CE',102)
    assert "OPTION_UNDERLYING_CONTEXT_MISSING" in caplog.text


def test_futures_snapshot_derives_slope_and_volume_ratio():
    st=_S(); ie=_IE(); m=StrategyManager([st], ie, _PM())
    ie.payload={'vwap':100,'close':100,'volume':1000,'avg_volume':1000}
    m.generate_signal('NFO:NIFTY26MAYFUT',100)
    ie.payload={'vwap':101,'close':102,'volume':2000,'avg_volume':1000}
    m.generate_signal('NFO:NIFTY26MAYFUT',102)
    snap = m._latest_context_snapshots.get('futures_context', {})
    assert float(snap.get('futures_volume_ratio') or 0.0) == 2.0
    assert float(snap.get('vwap_slope') or 0.0) > 0.0


def test_strategy_context_discards_stale_future() -> None:
    class _Hub:
        indicators_ready = True
        def get_active_futures_symbol(self):
            return "NFO:NIFTY26JUNFUT"
        def get_quote(self, *_args, **_kwargs):
            return {}

    st = _S(); ie = _IE(); m = StrategyManager([st], ie, _PM(), data_hub=_Hub())
    ie.payload = {'close': 110, 'previous_close': 100, 'volume': 100, 'avg_volume': 100}
    m.generate_signal('NSE:NIFTY', 110)
    ie.payload = {'close': 100, 'tick_slope': 0.2, 'volume': 100, 'avg_volume': 100}
    m.generate_signal('NFO:NIFTY26MAYFUT', 100)
    ie.payload = {'vwap': 102, 'exchange_vwap': 102, 'volume': 100, 'avg_volume': 100}
    m.generate_signal('NFO:NIFTY26JUN23900CE', 102)
    assert st.last.get('futures_context') is None
    assert 'stale_futures_context_discarded' in st.last.get('context_discard_reasons', [])
    assert st.last.get('spot_context')


def test_fresh_futures_context_can_supply_option_direction_when_spot_absent():
    st = _S(); ie = _IE(); m = StrategyManager([st], ie, _PM())
    ie.payload = {
        'close': 101,
        'previous_close': 100,
        'volume': 200,
        'avg_volume': 100,
        'futures_volume_ratio': 2.0,
        'tick_slope': 0.2,
    }
    m.generate_signal('NFO:NIFTY26MAYFUT', 101)
    ie.payload = {'vwap': 102, 'exchange_vwap': 102, 'volume': 100, 'avg_volume': 100}
    m.generate_signal('NFO:NIFTY26MAY23750CE', 102)
    assert st.last.get('direction_bias') == 'CE'
    assert st.last.get('underlying_direction_bias') == 'CE'
    assert isinstance(st.last.get('futures_context'), dict)
    assert 'spot_context' not in st.last


def test_runner_refreshes_context_symbols_before_option_evaluation():
    from types import SimpleNamespace
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    calls: list[tuple[str, float, str | None]] = []

    runner = object.__new__(StrategyRunner)
    runner._strategy_manager = SimpleNamespace(
        generate_signal=lambda symbol, price, trace_id=None: calls.append((symbol, price, trace_id))
    )
    runner._active_symbols = {'NSE:NIFTY', 'NFO:NIFTY26JUNFUT', 'NFO:NIFTY26JUN25000CE'}
    runner._suspended_context_symbol_until = {}
    runner._suspended_context_symbols = set()
    runner._logger = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    quotes = {
        'NSE:NIFTY': {'ltp': 25000.0},
        'NFO:NIFTY26JUNFUT': {'ltp': 25100.0},
    }
    runner.get_quote = lambda symbol: quotes.get(symbol)  # type: ignore[method-assign]

    runner._refresh_underlying_context_snapshots(trace_id='t1')

    assert ('NSE:NIFTY', 25000.0, 't1') in calls
    assert ('NFO:NIFTY26JUNFUT', 25100.0, 't1') in calls
    assert all(not symbol.endswith(('CE', 'PE')) for symbol, _price, _trace_id in calls)


def test_runner_context_symbols_bypass_trade_readiness_gate():
    from types import SimpleNamespace
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    decisions = []
    runner = object.__new__(StrategyRunner)
    runner._logger = SimpleNamespace(info=lambda *a, **k: decisions.append(k.get('extra', {})), error=lambda *a, **k: None)
    runner._is_context_symbol = lambda symbol: symbol == 'NSE:NIFTY'  # type: ignore[method-assign]

    assert runner._strategy_evaluation_allowed('NSE:NIFTY', trace_id='t1') is True
    assert decisions[-1]['reason'] == 'context_symbol_snapshot_update'
    assert decisions[-1]['trade_candidate'] is False


def test_runner_selected_option_depth_uses_ws_quote_proof_canonical_mapping():
    from types import SimpleNamespace
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    runner = object.__new__(StrategyRunner)
    runner._market_data = SimpleNamespace(
        has_ws_tradable_quote=lambda symbols: 'NFO:NIFTY26JUN25000CE' in symbols,
        time_since_last_tick=lambda symbol: 1.0,
    )
    runner._data_hub = None
    runner.get_quote = lambda symbol: None  # type: ignore[method-assign]

    assert runner._selected_option_has_real_depth('NIFTY26JUN25000CE') is True


def test_runner_selected_option_depth_requires_real_bid_ask():
    from types import SimpleNamespace
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    runner = object.__new__(StrategyRunner)
    runner._market_data = SimpleNamespace(has_ws_tradable_quote=lambda symbols: False)
    runner._data_hub = None
    runner.get_quote = lambda symbol: {'ltp': 100, 'bid': 99.5, 'ask': 100.5, 'tick_age_s': 1.0, 'depth': {'buy': [{'price': 99.5}], 'sell': [{'price': 100.5}]}}  # type: ignore[method-assign]

    assert runner._selected_option_has_real_depth('NFO:NIFTY26JUN25000CE') is True


def test_runner_refresh_includes_active_future_even_when_not_active_symbol():
    from types import SimpleNamespace
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    calls: list[str] = []
    runner = object.__new__(StrategyRunner)
    runner._strategy_manager = SimpleNamespace(
        generate_signal=lambda symbol, price, trace_id=None: calls.append(symbol)
    )
    runner._active_symbols = {'NSE:NIFTY', 'NFO:NIFTY26JUN25000CE'}
    runner._active_futures_symbol = 'NFO:NIFTY26JUNFUT'
    runner._suspended_context_symbol_until = {}
    runner._suspended_context_symbols = set()
    runner._market_data = SimpleNamespace(time_since_last_tick=lambda symbol: 1.0)
    runner._data_hub = None
    runner._logger = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    quotes = {
        'NSE:NIFTY': {'ltp': 25000.0, 'tick_age_s': 1.0},
        'NFO:NIFTY26JUNFUT': {'ltp': 25100.0, 'tick_age_s': 1.0},
    }
    runner.get_quote = lambda symbol: quotes.get(symbol)  # type: ignore[method-assign]

    runner._refresh_underlying_context_snapshots(trace_id='t2')

    assert calls == ['NSE:NIFTY', 'NFO:NIFTY26JUNFUT']


def test_runner_underlying_context_prefers_fresh_valid_spot_over_newer_future():
    from types import SimpleNamespace
    import time
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    now = time.time()
    runner = object.__new__(StrategyRunner)
    runner._strategy_manager = SimpleNamespace(
        _latest_context_snapshots={
            'spot_context': {'role': 'spot_context', 'timestamp': now - 5, 'direction_bias': 'CE', 'underlying_direction_confidence': 0.7},
            'futures_context': {'role': 'futures_context', 'timestamp': now - 1, 'direction_bias': 'PE', 'underlying_direction_confidence': 0.9},
        },
        _live_context_max_age_seconds=lambda: 60.0,
    )

    ctx = runner._underlying_context_from_strategy_manager()

    assert ctx['underlying_direction_bias'] == 'CE'
    assert ctx['direction_context_source'] == 'spot_context'
    assert 4.0 <= float(ctx['context_age_seconds']) <= 6.0


def test_runner_underlying_context_uses_future_when_spot_directionless():
    from types import SimpleNamespace
    import time
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    now = time.time()
    runner = object.__new__(StrategyRunner)
    runner._strategy_manager = SimpleNamespace(
        _latest_context_snapshots={
            'spot_context': {'role': 'spot_context', 'timestamp': now - 1, 'direction_bias': None},
            'futures_context': {'role': 'futures_context', 'timestamp': now - 2, 'direction_bias': 'PE', 'underlying_direction_confidence': 0.8},
        },
        _live_context_max_age_seconds=lambda: 60.0,
    )

    ctx = runner._underlying_context_from_strategy_manager()

    assert ctx['underlying_direction_bias'] == 'PE'
    assert ctx['direction_context_source'] == 'futures_context'
    assert ctx['spot_fresh'] is True
    assert ctx['futures_fresh'] is True


def test_runner_selected_option_depth_rejects_stale_cached_bid_ask():
    from types import SimpleNamespace
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    runner = object.__new__(StrategyRunner)
    runner._market_data = SimpleNamespace(has_ws_tradable_quote=lambda symbols: False, time_since_last_tick=lambda symbol: 120.0)
    runner._data_hub = None
    runner.get_quote = lambda symbol: {'ltp': 100, 'bid': 99.5, 'ask': 100.5, 'tick_age_s': 120.0}  # type: ignore[method-assign]

    assert runner._selected_option_has_real_depth('NFO:NIFTY26JUN25000CE') is False


def test_runner_selected_option_depth_rejects_cached_bid_ask_without_freshness():
    from types import SimpleNamespace
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    runner = object.__new__(StrategyRunner)
    runner._market_data = SimpleNamespace(has_ws_tradable_quote=lambda symbols: False)
    runner._data_hub = None
    runner.get_quote = lambda symbol: {'ltp': 100, 'bid': 99.5, 'ask': 100.5}  # type: ignore[method-assign]

    assert runner._selected_option_has_real_depth('NFO:NIFTY26JUN25000CE') is False
