from nifty_scalper_bot.execution.order_manager import OrderManager


class _Broker: pass
class _Pos: pass
class _Limiter: pass


def test_extract_quote_diagnostics_handles_malformed_fields():
    om = OrderManager(_Broker(), _Pos(), _Limiter())
    q = {'bid': 'bad', 'ask': None, 'ltp': '12.5', 'bid_qty': 'bad', 'timestamp': 'bad-ts'}
    out = om._extract_quote_diagnostics(q)
    assert out['bid'] == 0.0
    assert out['ask'] == 0.0
    assert out['ltp'] == 12.5
    assert out['bid_qty'] == 0
