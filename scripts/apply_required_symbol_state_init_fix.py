from pathlib import Path

path = Path("src/nifty_scalper_bot/data/market_data_manager.py")
text = path.read_text()
old = '''    def _required_live_symbols(self) -> set[str]:\n        """Return the canonical live-required symbol set for current readiness."""\n        now = time.monotonic()\n'''
new = '''    def _required_live_symbols(self) -> set[str]:\n        """Return the canonical live-required symbol set for current readiness."""\n        # Keep lifecycle bookkeeping owned by this method robust during early\n        # construction, lightweight diagnostics, and test doubles that bypass\n        # the full constructor. Production instances already initialise it.\n        if not isinstance(getattr(self, "_required_symbol_since_mono", None), dict):\n            self._required_symbol_since_mono = {}\n        now = time.monotonic()\n'''
if old not in text:
    raise RuntimeError("required live symbol method header not found")
path.write_text(text.replace(old, new, 1))
