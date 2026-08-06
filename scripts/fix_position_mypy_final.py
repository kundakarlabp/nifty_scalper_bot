from pathlib import Path
import re

path = Path("src/nifty_scalper_bot/execution/position_manager.py")
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    r"(?m)^(?P<indent>\s*)raw_value = row\.get\(column\)\n"
    r"(?P=indent)numeric = _safe_float\(raw_value\)$"
)
matches = list(pattern.finditer(text))
if len(matches) != 2:
    raise SystemExit(f"reconciliation numeric temporary count={len(matches)}")
names = iter(("raw_unrealized_value", "raw_realized_value"))


def replacement(match: re.Match[str]) -> str:
    name = next(names)
    indent = match.group("indent")
    return (
        f"{indent}{name} = row.get(column)\n"
        f"{indent}numeric = _safe_float({name})"
    )


text = pattern.sub(replacement, text)
path.write_text(text, encoding="utf-8")
