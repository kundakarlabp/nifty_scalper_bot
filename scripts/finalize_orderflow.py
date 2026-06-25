#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

path = Path("src/nifty_scalper_bot/strategies/elite_strategies/order_flow.py")
text = path.read_text(encoding="utf-8")

if "ORDERFLOW_REVERSAL_MAX_WINDOW_MS" not in text:
    pattern = re.compile(
        r'(\s+elapsed_ms = max\(0\.0, now - float\(state\.get\("started"\) or now\)\) \* 1000\.0\n)'
        r'(\s+return int\(state\.get\("count"\) or 0\) >= min_updates and elapsed_ms >= min_persistence_ms\n)'
    )
    replacement = (
        r'\1        max_window_ms = max(min_persistence_ms, safe_float_env("ORDERFLOW_REVERSAL_MAX_WINDOW_MS", 3000.0))\n'
        r'        if elapsed_ms > max_window_ms:\n'
        r'            state.update({"count": 1, "started": now, "version": version})\n'
        r'            return False\n\2'
    )
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit("OrderFlow max-window anchor missing")

text = text.replace(
    "fingerprint=(tick_age_ms, round(bid, 4), round(ask, 4), round(total_bid, 2), round(total_ask, 2))",
    "fingerprint=(round(bid, 4), round(ask, 4), round(total_bid, 2), round(total_ask, 2), tick_direction)",
    1,
)

if "'quote_readiness_allowed': quote_readiness.allowed" not in text:
    pattern = re.compile(
        r"(\s+'quote_update_version': quote_update_version,\n)"
        r"(\s+'selected_or_near_atm': selected_or_near_atm,\n)"
    )
    replacement = (
        r"\1                 'quote_readiness_allowed': quote_readiness.allowed,\n"
        r"                 'quote_readiness_reason': quote_readiness.reason,\n"
        r"                 'real_ticks_last_60s': quote_readiness.real_ticks_last_60s,\n"
        r"                 'real_tick_count_derived': quote_readiness.real_tick_count_derived,\n"
        r"                 'reversal_persistence_confirmed': reversal_persistence_confirmed,\n\2"
    )
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit("OrderFlow metadata anchor missing")

path.write_text(text, encoding="utf-8")
