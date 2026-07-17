from pathlib import Path
import re

path = Path('src/nifty_scalper_bot/data/market_data_manager.py')
text = path.read_text()

# 1. Required-symbol bookkeeping must be safe before the complete constructor path.
pattern = re.compile(
    r'(    def _required_live_symbols\(self\) -> set\[str\]:\n'
    r'(?:        """[^\n]*"""\n)?)'
)
match = pattern.search(text)
if match is None:
    raise SystemExit('required-live-symbols definition not found')
if 'required_since = getattr(self, "_required_symbol_since_mono", None)' not in text[match.start():match.start()+800]:
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

# 2. Read-only lifecycle classification must tolerate maps not yet initialized.
replacements = {
    '        token = self._symbol_to_token.get(canonical) or self._token_by_symbol.get(\n            canonical\n        )\n':
    '        symbol_to_token = getattr(self, "_symbol_to_token", {}) or {}\n        token_by_symbol = getattr(self, "_token_by_symbol", {}) or {}\n        token = symbol_to_token.get(canonical) or token_by_symbol.get(canonical)\n',
    '            sub_gen = self._symbol_subscription_generation.get(canonical)\n            tick_gen = self._symbol_first_tick_generation.get(canonical)\n            tick_mono = self._last_valid_live_tick_mono.get(canonical)\n            current_token = self._current_symbol_token_locked(canonical)\n            tracked = canonical in self._tracked_symbols\n':
    '            sub_gen = (getattr(self, "_symbol_subscription_generation", {}) or {}).get(canonical)\n            tick_gen = (getattr(self, "_symbol_first_tick_generation", {}) or {}).get(canonical)\n            tick_mono = (getattr(self, "_last_valid_live_tick_mono", {}) or {}).get(canonical)\n            current_token = self._current_symbol_token_locked(canonical)\n            tracked = canonical in (getattr(self, "_tracked_symbols", set()) or set())\n',
}
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit(f'expected lifecycle anchor missing: {old[:80]!r}')

path.write_text(text)
