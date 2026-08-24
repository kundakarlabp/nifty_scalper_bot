from pathlib import Path

path = Path("src/nifty_scalper_bot/execution/quote_readiness.py")
text = path.read_text(encoding="utf-8")
old = 'timestamp_ms = _float(payload, "last_tick_ts_ms", "timestamp_ms")'
new = 'timestamp_ms = _float(payload, "last_tick_ts_ms")'
if text.count(old) != 1:
    raise SystemExit(f"expected one timestamp precedence site, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
