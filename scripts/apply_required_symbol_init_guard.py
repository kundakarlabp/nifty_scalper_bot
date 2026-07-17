from pathlib import Path

path = Path('src/nifty_scalper_bot/data/market_data_manager.py')
text = path.read_text()
needle = '''    def _required_live_symbols(self) -> set[str]:\n        """Return canonical symbols required for live trading health checks."""\n'''
replacement = '''    def _required_live_symbols(self) -> set[str]:\n        """Return canonical symbols required for live trading health checks."""\n        # Keep the ownership boundary safe for early lifecycle diagnostics and\n        # lightweight instances created without the full constructor. Production\n        # instances already initialize this mapping in __init__.\n        required_since = getattr(self, "_required_symbol_since_mono", None)\n        if not isinstance(required_since, dict):\n            required_since = {}\n            self._required_symbol_since_mono = required_since\n'''
if needle not in text:
    raise SystemExit('required-live-symbols anchor not found or already changed')
path.write_text(text.replace(needle, replacement, 1))

test_path = Path('tests/data/test_canonical_history_hydration.py')
test_text = test_path.read_text()
marker = 'def test_feed_health_uses_selected_options_not_context_option_staleness'
if marker not in test_text:
    raise SystemExit('feed-health regression anchor missing')
addition = '''\n\ndef test_required_live_symbols_self_initializes_lifecycle_bookkeeping() -> None:\n    mdm = MarketDataManager.__new__(MarketDataManager)\n    mdm._readiness_requirements = {}\n    mdm._active_subscribed_symbols = set()\n    mdm._tracked_symbols = set()\n\n    assert mdm._required_live_symbols() == set()\n    assert mdm._required_symbol_since_mono == {}\n'''
if 'test_required_live_symbols_self_initializes_lifecycle_bookkeeping' not in test_text:
    test_path.write_text(test_text + addition)
