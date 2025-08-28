from src.data.source import LiveKiteSource


class DummyKite:
    def __init__(self):
        self.last_instruments = None

    def ltp(self, instruments):
        self.last_instruments = instruments
        return {"NSE:NIFTY 50": {"last_price": 200.0}}


def test_get_last_price_normalizes_nifty_symbol():
    src = LiveKiteSource(kite=DummyKite())
    price = src.get_last_price("NIFTY50")
    assert price == 200.0
    # ensure the kite client received the normalized symbol
    assert src.kite.last_instruments == ["NSE:NIFTY 50"]
