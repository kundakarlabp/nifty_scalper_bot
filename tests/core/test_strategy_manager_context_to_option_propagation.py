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
