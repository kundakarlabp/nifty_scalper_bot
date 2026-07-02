from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/nifty_scalper_bot/core/app.py"
text = path.read_text(encoding="utf-8")
field_name = "canonical_history_ensurer_injection_failed"

registry_anchor = "BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS: dict[str, Any] = {\n"
registry_line = f'    "{field_name}": False,\n'
if registry_line not in text:
    if text.count(registry_anchor) != 1:
        raise RuntimeError("BotContext runtime defaults anchor mismatch")
    text = text.replace(registry_anchor, registry_anchor + registry_line, 1)

field_line = f"    {field_name}: bool = False\n"
if field_line not in text:
    class_anchor = "class BotContext:\n"
    class_start = text.find(class_anchor)
    if class_start < 0:
        raise RuntimeError("BotContext class anchor missing")
    body_start = class_start + len(class_anchor)
    next_top_level = text.find("\nclass ", body_start)
    next_dataclass = text.find("\n@dataclass", body_start)
    class_end_candidates = [i for i in (next_top_level, next_dataclass) if i >= 0]
    class_end = min(class_end_candidates) if class_end_candidates else len(text)
    block = text[body_start:class_end]
    lines = block.splitlines(keepends=True)
    insert_offset = None
    cursor = 0
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 4 and (
            stripped.startswith("def ")
            or stripped.startswith("async def ")
            or stripped.startswith("@")
        ):
            insert_offset = cursor
            break
        cursor += len(line)
    if insert_offset is None:
        raise RuntimeError("BotContext method boundary missing")
    insert_at = body_start + insert_offset
    text = text[:insert_at] + field_line + text[insert_at:]

basket_anchor = '''    basket_copy = dict(basket or {})
    basket_copy["futures_symbol"] = active_futures_symbol
    basket = normalize_active_basket_schema(basket_copy)
'''
basket_replacement = '''    basket_copy = dict(basket or {})
    basket_copy["futures_symbol"] = active_futures_symbol
    basket_copy["option_symbols"] = [
        str(sym) for sym in option_symbols if str(sym).endswith(("CE", "PE"))
    ]
    basket_copy["symbols"] = [str(sym) for sym in symbols if sym]
    if atm_strike is not None:
        basket_copy["atm_strike"] = atm_strike
    basket = normalize_active_basket_schema(basket_copy)
'''
if text.count(basket_anchor) != 1:
    raise RuntimeError("active basket pre-normalization anchor mismatch")
text = text.replace(basket_anchor, basket_replacement, 1)

path.write_text(text, encoding="utf-8")
