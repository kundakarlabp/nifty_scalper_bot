from src.data.source import LiveKiteSource

class DummyKite:
    def ltp(self, instruments):
        return {}

def test_get_last_price_missing_symbol_logs_warning(caplog):
    src = LiveKiteSource(kite=DummyKite())
    with caplog.at_level("WARNING"):
        out = src.get_last_price("NSE:FOO")
    assert out is None
    assert "NSE:FOO" in caplog.text and "not found" in caplog.text
