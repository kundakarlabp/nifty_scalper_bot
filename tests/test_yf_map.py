from data.yf_fallback import _map_symbol


def test_yfinance_symbol_map() -> None:
    assert _map_symbol("NIFTY 50") == "^NSEI"
    assert _map_symbol("NIFTY BANK") == "^NSEBANK"
