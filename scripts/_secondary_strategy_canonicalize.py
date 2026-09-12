from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    target = Path(path)
    text = target.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected one source seam, found {text.count(old)}")
    target.write_text(text.replace(old, new, 1))


# OrderFlow owns quote identity natively and remains context-only.
path = "src/nifty_scalper_bot/strategies/elite_strategies/order_flow.py"
replace_once(
    path,
    "import time\nfrom typing import Any\n",
    "import time\nimport zlib\nfrom typing import Any, Mapping\n",
    "order_flow imports",
)
replace_once(
    path,
    "from nifty_scalper_bot.strategies.elite_strategies.order_flow_live_context_patch import (\n"
    "    apply_orderflow_live_context_proof,\n"
    ")\n",
    "",
    "order_flow sidecar import",
)
helper_anchor = "    return support, strong\n\n\nclass OrderFlowStrategy"
helper = '''    return support, strong


def _safe_float_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _stable_quote_version(value: Any) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = 0
    if numeric > 0:
        return numeric
    return int(zlib.crc32(str(value).encode("utf-8")) & 0x7FFFFFFF) or 1


def _stamp_quote_update_identity(
    metadata: dict[str, Any], indicators: Mapping[str, Any]
) -> None:
    """Preserve a real quote version or stable microstructure fingerprint."""
    for source in (metadata, indicators):
        for key in (
            "quote_update_version",
            "update_version",
            "tick_version",
            "last_tick_ts_ms",
            "timestamp_ms",
            "last_tick_timestamp",
        ):
            value = source.get(key)
            if value not in (None, "", 0, 0.0):
                metadata["quote_update_version"] = _stable_quote_version(value)
                metadata.setdefault("quote_update_version_source", key)
                return

    bid = _safe_float_value(metadata.get("bid") or indicators.get("bid"))
    ask = _safe_float_value(metadata.get("ask") or indicators.get("ask"))
    imbalance = _safe_float_value(
        metadata.get("depth_imbalance") or indicators.get("depth_imbalance")
    )
    tick_direction = str(
        metadata.get("tick_direction") or indicators.get("tick_direction") or ""
    ).upper()
    if bid is None and ask is None and imbalance is None and not tick_direction:
        return
    raw = (
        f"{bid if bid is not None else 'na'}:"
        f"{ask if ask is not None else 'na'}:"
        f"{imbalance if imbalance is not None else 'na'}:{tick_direction or 'na'}"
    )
    metadata["quote_update_version"] = _stable_quote_version(raw)
    metadata["quote_update_version_source"] = "microstructure_fingerprint"


class OrderFlowStrategy'''
replace_once(path, helper_anchor, helper, "order_flow quote helper")
replace_once(
    path,
    "            return apply_orderflow_live_context_proof(signal, indicators)",
    "            _stamp_quote_update_identity(signal.metadata, indicators)\n"
    "            return signal",
    "order_flow sidecar call",
)
Path(
    "src/nifty_scalper_bot/strategies/elite_strategies/order_flow_live_context_patch.py"
).unlink()

# Tuesday gamma: contract expiry/calendar semantics; underlying structure stays in spot domain.
path = "src/nifty_scalper_bot/strategies/elite_tuesday_gamma_buyer.py"
replace_once(
    path,
    "from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom datetime import date\nimport os\n",
    "from __future__ import annotations\n\nimport os\nfrom dataclasses import dataclass\nfrom datetime import date\n",
    "Tuesday gamma import order",
)
replace_once(
    path,
    "from nifty_scalper_bot.utils.logging import get_logger\n",
    "from nifty_scalper_bot.utils.logging import get_logger\n"
    "from nifty_scalper_bot.utils.smart_symbol import (\n"
    "    WEEKLY_EXPIRY_WEEKDAY,\n"
    "    get_actual_expiry_date,\n"
    ")\n",
    "Tuesday gamma expiry imports",
)
replace_once(
    path,
    "            if now_local is not None:\n"
    "                if now_local.weekday() != 1:\n"
    "                    return None\n"
    "                if now_local.hour < 9 or now_local.hour > 14:\n"
    "                    return None\n"
    "                if now_local.hour == 9 and now_local.minute < 20:\n"
    "                    return None\n"
    "                if now_local.hour == 14 and now_local.minute >= 45:\n"
    "                    return None\n",
    "            if now_local is not None:\n"
    "                raw_days_to_expiry = indicators.get('days_to_expiry')\n"
    "                if raw_days_to_expiry is not None:\n"
    "                    try:\n"
    "                        expiry_session = float(raw_days_to_expiry) <= 0.0\n"
    "                    except (TypeError, ValueError):\n"
    "                        expiry_session = False\n"
    "                else:\n"
    "                    today = now_local.date()\n"
    "                    expiry_session = (\n"
    "                        get_actual_expiry_date(today, WEEKLY_EXPIRY_WEEKDAY) == today\n"
    "                    )\n"
    "                if not expiry_session:\n"
    "                    return None\n"
    "                if now_local.hour < 9 or now_local.hour > 14:\n"
    "                    return None\n"
    "                if now_local.hour == 9 and now_local.minute < 20:\n"
    "                    return None\n"
    "                if now_local.hour == 14 and now_local.minute >= 45:\n"
    "                    return None\n",
    "Tuesday gamma expiry gate",
)
replace_once(
    path,
    "            if current_price > recent_high:\n"
    "                score += 2\n"
    "            elif recent_low > 0 and current_price < recent_low:\n"
    "                score -= 2\n",
    "            if recent_high > 0 and spot > recent_high:\n"
    "                score += 2\n"
    "            elif recent_low > 0 and spot < recent_low:\n"
    "                score -= 2\n",
    "Tuesday gamma price domain",
)

# Obsolete environment switches must not imply that OrderFlow can trigger.
env = Path(".env.example")
env_text = env.read_text()
for obsolete in (
    "ORDERFLOW_ALLOW_TRIGGER_ROLE=true\n",
    "ORDERFLOW_ALLOW_LIVE_TRIGGER=true\n",
    "ORDERFLOW_ALLOW_TRIGGER_WITHOUT_DIRECTION_LIVE=false\n",
):
    if obsolete not in env_text:
        raise SystemExit(f"missing expected obsolete env line: {obsolete.strip()}")
    env_text = env_text.replace(obsolete, "", 1)
env.write_text(env_text)

# Quote-identity tests follow the canonical OrderFlow owner.
path = "tests/strategies/test_signal_observability_contract.py"
replace_once(
    path,
    "from nifty_scalper_bot.strategies.elite_strategies.order_flow_live_context_patch import (\n"
    "    apply_orderflow_live_context_proof,\n"
    ")\n",
    "from nifty_scalper_bot.strategies.elite_strategies.order_flow import (\n"
    "    _stamp_quote_update_identity,\n"
    ")\n",
    "observability import",
)
replace_once(
    path,
    "    first = apply_orderflow_live_context_proof(_orderflow_signal(), {})\n"
    "    repeated = apply_orderflow_live_context_proof(_orderflow_signal(), {})\n"
    "    changed = apply_orderflow_live_context_proof(_orderflow_signal(bid=99.6), {})\n\n",
    "    first = _orderflow_signal()\n"
    "    repeated = _orderflow_signal()\n"
    "    changed = _orderflow_signal(bid=99.6)\n"
    "    _stamp_quote_update_identity(first.metadata, {})\n"
    "    _stamp_quote_update_identity(repeated.metadata, {})\n"
    "    _stamp_quote_update_identity(changed.metadata, {})\n\n",
    "observability fingerprint calls",
)
