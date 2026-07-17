from pathlib import Path
import re

path = Path('src/nifty_scalper_bot/data/market_data_manager.py')
text = path.read_text()
pattern = re.compile(
    r'(    def _required_live_symbols\(self\) -> set\[str\]:\n'
    r'(?:        """[^\n]*"""\n)?)'
)
match = pattern.search(text)
if match is None:
    raise SystemExit('required-live-symbols definition not found')
if '_required_live_symbols(self) -> set[str]:' in text and 'required_since = getattr(self, "_required_symbol_since_mono", None)' not in text[match.start():match.start()+800]:
    guard = (
        '        # Keep this ownership boundary safe during early lifecycle diagnostics\n'
        '        # and lightweight construction. Normal production instances already\n'
        '        # initialize the mapping in __init__.\n'
        '        required_since = getattr(self, "_required_symbol_since_mono", None)\n'
        '        if not isinstance(required_since, dict):\n'
        '            required_since = {}\n'
        '            self._required_symbol_since_mono = required_since\n'
    )
    text = text[:match.end()] + guard + text[match.end():]
    path.write_text(text)

test_path = Path('tests/data/test_canonical_history_hydration.py')
test_text = test_path.read_text()
marker = 'def test_feed_health_uses_selected_options_not_context_option_staleness'
if marker not in test_text:
    raise SystemExit('feed-health regression anchor missing')
addition = '''\n\ndef test_required_live_symbols_self_initializes_lifecycle_bookkeeping() -> None:\n    mdm = MarketDataManager.__new__(MarketDataManager)\n    mdm._readiness_requirements = {}\n    mdm._active_subscribed_symbols = set()\n    mdm._tracked_symbols = set()\n\n    assert mdm._required_live_symbols() == set()\n    assert mdm._required_symbol_since_mono == {}\n'''
if 'test_required_live_symbols_self_initializes_lifecycle_bookkeeping' not in test_text:
    test_path.write_text(test_text + addition)
