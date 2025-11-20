from datetime import date, datetime
from nifty_scalper_bot.core import app
from nifty_scalper_bot.utils import smart_symbol

def test_get_symbols_returns_valid_nfo_prefixed(monkeypatch):
    # monkeypatch a minimal resolver mapping
    sample_inst = {
        "NFO:NIFTY25N25124000CE": {"tradingsymbol": "NIFTY25N25124000CE", "instrument_token": 12345}
    }
    monkeypatch.setitem(globals(), "instrument_resolver", None)  # ensure none in globals
    # patch resolver to be a simple dict object with 'symbols' attribute
    class FakeResolver:
        symbols = sample_inst
        def lookup(self, k): return sample_inst.get(k)
    monkeypatch.setitem(globals(), "instrument_resolver", FakeResolver())
    # patch smart_symbol.get_next_valid_symbols to return the sample entry
    def fake_next_valid(strikes, opt_types, instrument_map):
        return [sample_inst["NFO:NIFTY25N25124000CE"]]
    monkeypatch.setattr(smart_symbol, "get_next_valid_symbols", fake_next_valid)
    cfg = type("C", (), {"symbols": None})
    syms = app._get_symbols(cfg)
    assert any(s.startswith("NFO:NIFTY") for s in syms)
