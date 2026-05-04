import inspect

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.data.market_data_policy import MarketDataPolicy


def test_quote_bulk_policy_defined_before_aliases(monkeypatch):
    src = inspect.getsource(ZerodhaKiteClient.get_quote_bulk)
    assert src.index('policy =') < src.index('policy.quote_aliases')
    client = ZerodhaKiteClient.__new__(ZerodhaKiteClient)
    client._md_policy = MarketDataPolicy.from_env()
    client._QUOTE_BUCKET = 'q'
    client._acquire_bucket = lambda *_: None
    client._ensure_json = lambda x: x
    client._make_request = lambda *a, **k: {'data': {'NSE:NIFTY': {'last_price': 1}}}
    out = client.get_quote_bulk(['NSE:NIFTY'])
    assert isinstance(out, dict)
