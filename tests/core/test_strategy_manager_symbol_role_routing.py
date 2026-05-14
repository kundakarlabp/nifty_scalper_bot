from __future__ import annotations
import typing as t
from nifty_scalper_bot.core.strategy_manager import StrategyManager


class _S:
    def __init__(self, name:str='SMC') -> None:
        self.name=name
        self.calls=0
        self.last_indicators: dict[str,t.Any] = {}
    def get_required_indicators(self)->list[str]:
        return []
    def generate_signal(self, symbol:str, indicators:t.Mapping[str,t.Any], current_price:float, position:t.Any):
        self.calls+=1
        self.last_indicators=dict(indicators)
        return None


class _IE:
    def get_history(self, symbol:str):
        return [1]*50
    def get_indicators(self, symbol:str, names:t.Iterable[str]):
        return {'vwap':100.0,'exchange_vwap':100.0,'volume':1000.0,'avg_volume':900.0,'direction_bias':'CE'}

class _PM:
    def get_position(self, symbol:str):
        return None


def test_context_spot_skips_strategy_and_updates_snapshot(caplog):
    st=_S()
    m=StrategyManager([st], _IE(), _PM())
    assert m.generate_signal('NSE:NIFTY',100.0) is None
    assert st.calls==0
    assert m._latest_context_snapshots['spot_context']['symbol']=='NSE:NIFTY'
    assert any('CONTEXT_SYMBOL_STRATEGY_EVAL_SKIPPED' in r.message for r in caplog.records)


def test_context_futures_skips_strategy_and_updates_snapshot():
    st=_S()
    m=StrategyManager([st], _IE(), _PM())
    assert m.generate_signal('NFO:NIFTY26MAYFUT',100.0) is None
    assert st.calls==0
    assert m._latest_context_snapshots['futures_context']['role']=='futures_context'


def test_option_runs_strategy_and_receives_context():
    st=_S('VWAPPro')
    m=StrategyManager([st], _IE(), _PM())
    m.generate_signal('NSE:NIFTY',100.0)
    m.generate_signal('NFO:NIFTY26MAYFUT',100.0)
    m.generate_signal('NFO:NIFTY26MAY23500CE',100.0)
    assert st.calls==1
    assert 'spot_context' in st.last_indicators
    assert 'futures_context' in st.last_indicators
