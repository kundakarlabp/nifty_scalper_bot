from pathlib import Path


def test_spot_and_futures_share_history_backed_direction_evidence():
    source = Path("src/nifty_scalper_bot/core/strategy_manager.py").read_text()
    assert source.count('role not in {"spot_context", "futures_context"}') >= 2
    assert 'if role in {"spot_context", "futures_context"}:\n            if vwap_slope is None:' in source
    assert 'if role == "futures_context":\n            if vwap_slope is None:' not in source


def test_option_history_is_not_promoted_to_underlying_direction_authority():
    source = Path("src/nifty_scalper_bot/core/strategy_manager.py").read_text()
    assert 'OPTION PREMIUM DATA MUST NEVER AUTHORIZE UNDERLYING DIRECTION' in source
    assert 'if role not in {"spot_context", "futures_context"}' in source
