from pathlib import Path

def test_required_bars_defined_once() -> None:
    src = Path('src/nifty_scalper_bot/strategies/runner.py').read_text()
    assert src.count('def _required_bars_for_symbol') == 1
