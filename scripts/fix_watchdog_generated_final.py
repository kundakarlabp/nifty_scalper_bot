from pathlib import Path

path = Path("src/nifty_scalper_bot/infra/watchdog.py")
text = path.read_text(encoding="utf-8")
old = 'or presssure.get("one_tick_p99_ms")'
new = 'or pressure.get("one_tick_p99_ms")'
count = text.count(old)
if count != 1:
    raise SystemExit(f"watchdog pressure typo count={count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
