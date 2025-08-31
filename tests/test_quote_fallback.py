from src.execution.order_executor import fetch_quote_with_depth


class DummyKite:
    def quote(self, *args, **kwargs):
        raise Exception("quote fail")

    def ltp(self, symbols):
        return {symbols[0]: {"last_price": 123.0}}


def test_fetch_quote_with_depth_uses_ltp_on_failure():
    kite = DummyKite()
    q = fetch_quote_with_depth(kite, "NIFTY24APR10000CE")
    assert q["ltp"] == 123.0
    assert q["bid"] == 0.0 and q["ask"] == 0.0

