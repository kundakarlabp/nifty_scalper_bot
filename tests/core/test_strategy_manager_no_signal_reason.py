from __future__ import annotations
import typing as t
from nifty_scalper_bot.core.strategy_manager import StrategyManager

class _S:
    name='SMC'
    def get_required_indicators(self): return []
    def generate_signal(self, symbol:str, indicators:t.Mapping[str,t.Any], current_price:float, position:t.Any):
        return None

class _IE:
    def get_history(self, symbol:str): return [1]*30
    def get_indicators(self, symbol:str, names:t.Iterable[str]):
        return {'vwap':100.0,'exchange_vwap':100.0,'volume':10.0,'avg_volume':10.0}

class _PM:
    def get_position(self, symbol:str): return None


def test_no_signal_reason_not_data_invalid(caplog):
    m=StrategyManager([_S()], _IE(), _PM())
    assert m.generate_signal('NFO:NIFTY26MAY23500CE',100.0) is None
    assert any('reason=no_strategy_signal' in r.message or (getattr(r,'reason_code',None)=='no_strategy_signal') for r in caplog.records)
