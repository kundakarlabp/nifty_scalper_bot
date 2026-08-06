from pathlib import Path

path = Path("src/nifty_scalper_bot/infra/watchdog.py")
text = path.read_text(encoding="utf-8")
count = text.count("presssure")
if count < 1:
    raise SystemExit("watchdog pressure typo not found")
text = text.replace("presssure", "pressure")
if "presssure" in text:
    raise SystemExit("watchdog pressure typo remains")
path.write_text(text, encoding="utf-8")
