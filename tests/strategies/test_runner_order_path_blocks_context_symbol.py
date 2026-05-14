from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
import threading
from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal

class _L:
    def debug(self,*a,**k): pass
    def info(self,*a,**k): pass
    def warning(self,*a,**k): pass
    def error(self,*a,**k): pass
    def critical(self,*a,**k): pass

def test_live_path_blocks_non_option_symbol():
    r=StrategyRunner.__new__(StrategyRunner)
    r._logger=_L(); r._order_manager=object(); r._runtime_live_orders_armed=True
    r._underlying_last_signal_ts={}; r._reason_last_signal_ts={}; r._premium_squeeze_last_signal_ts={}
    r._underlying_signal_cooldown_seconds=0.0; r._reason_signal_cooldown_seconds=0.0
    r._cooldown_log_throttle_seconds=1.0; r._order_attempt_window=deque(); r._max_order_attempts_per_minute=10
    r._reset_execution_state=lambda *_a,**_k: None; r._entry_lock=threading.Lock(); r._signal_reject_cooldown_ts={}
    r._extract_underlying=lambda s:s
    s=Signal(action='BUY',symbol='NSE:NIFTY',quantity=1,confidence=1.0,reason='x',stop_loss=1,take_profit=2,metadata={})
    out=r._handle_entry_signal_inner(s,'NSE:NIFTY','NSE:NIFTY',100.0,datetime.now(timezone.utc),trace_id='t')
    assert out.reason == 'non_option_symbol'
