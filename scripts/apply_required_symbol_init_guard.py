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
    path.write_text(text)
