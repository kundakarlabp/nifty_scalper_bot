import pytest
from market_data_manager import MarketDataManager

class DummyBroker:
    def get_quote(self, symbol):
        return None

def make_mdm():
    mdm = MarketDataManager(broker=DummyBroker(), ws=None, resolver=None, settings=None)
    return mdm

def test_refresh_quote_now_no_quote_returns_none():
    mdm = make_mdm()
    result = mdm.refresh_quote_now("FAKE:SYM", trace_id="t1")
    assert result is None

def test_probe_quote_contains_diagnostics():
    mdm = make_mdm()
    report = mdm.probe_quote("FAKE:SYM")
    assert isinstance(report, dict)
    # keys we expect
    assert "cache" in report
    assert "normalized" in report
    norm = report["normalized"]
    assert "has_depth" in norm
    assert "source" in norm
