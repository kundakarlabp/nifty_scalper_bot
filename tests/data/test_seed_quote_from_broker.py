import logging

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class Broker:
    def get_quote(self, symbol: str):
        return {'last_price': 100.0, 'bid': 99.5, 'ask': 100.5, 'instrument_token': 123}


def test_seed_quote_emits_rest_tick() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = logging.getLogger('test_mdm')
    mdm._lock = type('L', (), {'__enter__': lambda s: s, '__exit__': lambda s,a,b,c: None})()
    mdm._broker = Broker()
    mdm._seeded_symbols = set()
    mdm._seed_completed = False
    mdm._latest_ticks = {}
    mdm._last_tick_source = {}
    mdm._is_ws_healthy = lambda: False
    mdm._canonical_symbol = lambda s: s

    emitted = {}

    def emit(symbol, tick, source):
        emitted.update({'symbol': symbol, 'tick': tick, 'source': source})

    mdm._emit_tick = emit
    ok = mdm._seed_quote_from_broker('NSE:NIFTY')
    assert ok is True
    assert emitted['source'] == 'rest'
    assert emitted['tick']['ltp'] == 100.0
    assert emitted['tick']['last_price'] == 100.0
    assert isinstance(emitted['tick']['timestamp'], str)
    assert 'received_at' in emitted['tick']
