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
