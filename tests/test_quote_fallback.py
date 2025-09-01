from src.execution import order_executor as oe


class DummyKite:
    def quote(self, *args, **kwargs):
        raise Exception("quote fail")

    def ltp(self, symbols):
        return {symbols[0]: {"last_price": 123.0}}


def test_fetch_quote_with_depth_uses_ltp_on_failure():
    oe._QUOTE_CACHE.clear()
    kite = DummyKite()
    q = oe.fetch_quote_with_depth(kite, "NIFTY24APR10000CE")
    assert q["ltp"] == 123.0
    assert q["source"] == "ltp_fallback"
    assert q["bid"] < q["ltp"] < q["ask"]


class DummyKiteQuote:
    def quote(self, symbols):
        return {
            symbols[0]: {
                "last_price": 101.0,
                "depth": {
                    "buy": [{"price": 100.0, "quantity": 1}],
                    "sell": [{"price": 102.0, "quantity": 1}],
                },
            }
        }


class DummyKiteFail:
    def quote(self, *args, **kwargs):  # pragma: no cover - simple exception
        raise Exception("quote fail")

    def ltp(self, *args, **kwargs):  # pragma: no cover - no data
        raise Exception("ltp fail")


def test_fetch_quote_with_depth_uses_cache_when_quote_missing():
    oe._QUOTE_CACHE.clear()
    tsym = "NIFTY24APR10100CE"
    good = DummyKiteQuote()
    first = oe.fetch_quote_with_depth(good, tsym)
    assert first["ltp"] == 101.0
    bad = DummyKiteFail()
    cached = oe.fetch_quote_with_depth(bad, tsym)
    assert cached["ltp"] == 101.0
    assert cached["bid"] == 100.0
    assert cached["ask"] == 102.0
    assert cached["source"] == "cache"

