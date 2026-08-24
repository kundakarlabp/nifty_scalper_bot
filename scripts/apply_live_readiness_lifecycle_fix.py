from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# 1) Canonical reconciliation freshness: reuse the existing max-age policy in
#    health, self-check, runtime arming, and the re-arm cache fast path.
# ---------------------------------------------------------------------------
path = "src/nifty_scalper_bot/core/app.py"
text = read(path)
text = replace_once(
    text,
    '''    return max(0.0, value)\n\n\ndef get_http_app() -> FastAPI:\n''',
    '''    return max(0.0, value)\n\n\ndef _reconciliation_is_fresh(ctx: Any) -> bool:\n    \"\"\"Return whether the last successful broker reconciliation is still valid.\"\"\"\n    if not hasattr(ctx, \"position_reconciliation_completed\"):\n        return True\n    if not bool(getattr(ctx, \"position_reconciliation_completed\", False)):\n        return False\n    max_age_s = _reconciliation_max_age_seconds()\n    if max_age_s <= 0:\n        return True\n    completed_at = getattr(ctx, \"position_reconciliation_completed_at\", None)\n    if completed_at is None:\n        return False\n    try:\n        age_s = (datetime.now(timezone.utc) - completed_at).total_seconds()\n    except Exception:  # noqa: BLE001 - malformed lifecycle state fails closed\n        return False\n    return age_s <= max_age_s\n\n\ndef get_http_app() -> FastAPI:\n''',
    label="insert reconciliation freshness helper",
)
text = replace_once(
    text,
    '''        if not bool(getattr(ctx, \"position_reconciliation_completed\", False)):\n            blockers.append(\"position_reconciliation_incomplete\")\n        else:\n            # `completed` now survives an in-flight refresh (last-known-good),\n            # so it must be age-bounded: a permanently stuck or failing\n            # reconciler must not leave execution armed indefinitely.\n            _completed_at = getattr(ctx, \"position_reconciliation_completed_at\", None)\n            _max_age_s = _reconciliation_max_age_seconds()\n            if _max_age_s > 0:\n                if _completed_at is None:\n                    blockers.append(\"position_reconciliation_stale\")\n                else:\n                    try:\n                        _age = (\n                            datetime.now(timezone.utc) - _completed_at\n                        ).total_seconds()\n                    except Exception:  # noqa: BLE001 - fail closed on bad state\n                        _age = None\n                    if _age is None or _age > _max_age_s:\n                        blockers.append(\"position_reconciliation_stale\")\n''',
    '''        if not bool(getattr(ctx, \"position_reconciliation_completed\", False)):\n            blockers.append(\"position_reconciliation_incomplete\")\n        elif not _reconciliation_is_fresh(ctx):\n            blockers.append(\"position_reconciliation_stale\")\n''',
    label="reuse freshness helper in HTTP trading health",
)
text = replace_once(
    text,
    '''        if not bool(getattr(self._context, \"position_reconciliation_completed\", False)):\n            return (\n                False,\n                \"position_reconciliation_incomplete\",\n                {\"blocker\": \"position_reconciliation_incomplete\"},\n            )\n        if bool(getattr(self._context, \"unprotected_broker_positions\", set())):\n''',
    '''        if not bool(getattr(self._context, \"position_reconciliation_completed\", False)):\n            return (\n                False,\n                \"position_reconciliation_incomplete\",\n                {\"blocker\": \"position_reconciliation_incomplete\"},\n            )\n        if not _reconciliation_is_fresh(self._context):\n            return (\n                False,\n                \"position_reconciliation_stale\",\n                {\"blocker\": \"position_reconciliation_stale\"},\n            )\n        if bool(getattr(self._context, \"unprotected_broker_positions\", set())):\n''',
    label="self-check reconciliation freshness",
)
text = replace_once(
    text,
    '''                and not bool(getattr(ctx, \"position_reconciliation_failed\", False))\n                and not bool(getattr(ctx, \"broker_auth_invalid\", False))\n''',
    '''                and not bool(getattr(ctx, \"position_reconciliation_failed\", False))\n                and _reconciliation_is_fresh(ctx)\n                and not bool(getattr(ctx, \"broker_auth_invalid\", False))\n''',
    label="rearm fast path reconciliation freshness",
)
text = replace_once(
    text,
    '''        if (\n            live_mode\n            and hasattr(ctx, \"position_reconciliation_completed\")\n            and not bool(getattr(ctx, \"position_reconciliation_completed\", False))\n        ):\n            missing.append(\"position_reconciliation_incomplete\")\n        bracket_manager = getattr(ctx, \"bracket_manager\", None)\n''',
    '''        if (\n            live_mode\n            and hasattr(ctx, \"position_reconciliation_completed\")\n            and not bool(getattr(ctx, \"position_reconciliation_completed\", False))\n        ):\n            missing.append(\"position_reconciliation_incomplete\")\n        elif (\n            live_mode\n            and hasattr(ctx, \"position_reconciliation_completed\")\n            and not _reconciliation_is_fresh(ctx)\n        ):\n            missing.append(\"position_reconciliation_stale\")\n        bracket_manager = getattr(ctx, \"bracket_manager\", None)\n''',
    label="canonical readiness reconciliation stale blocker",
)
text = replace_once(
    text,
    '''        and (\n            not hasattr(ctx, \"position_reconciliation_completed\")\n            or bool(getattr(ctx, \"position_reconciliation_completed\", False))\n        ),\n''',
    '''        and (\n            not hasattr(ctx, \"position_reconciliation_completed\")\n            or _reconciliation_is_fresh(ctx)\n        ),\n''',
    label="canonical execution-ready reconciliation freshness",
)
write(path, text)


# ---------------------------------------------------------------------------
# 2) Broker session health: reuse an authenticated funds REST call as the
#    existing session proof. Keep order_endpoint_verified diagnostic-only and
#    keep reconciliation completion independent from broker endpoint labels.
# ---------------------------------------------------------------------------
path = "src/nifty_scalper_bot/main.py"
text = read(path)
text = replace_once(
    text,
    '''    elif funds_endpoint_verified:\n        state = \"funds_verified\"\n        authentication_known = False\n''',
    '''    elif funds_endpoint_verified:\n        state = \"funds_verified\"\n        authentication_known = True\n''',
    label="funds endpoint proves authenticated session",
)
text = replace_once(
    text,
    '''        \"funds_verified\": \"unknown\",\n''',
    '''        \"funds_verified\": \"authenticated\",\n''',
    label="structured funds auth mapping",
)
text = replace_once(
    text,
    '''        broker_status[\"order_endpoint_verified\"]\n        and reconciliation_completed\n''',
    '''        broker_status[\"funds_endpoint_verified\"]\n        and reconciliation_completed\n''',
    label="live order readiness uses proven authenticated REST session",
)
text = replace_once(
    text,
    '''    if not broker_status[\"order_endpoint_verified\"]:\n        missing.append(\"order_endpoint_unverified\")\n''',
    '''    if not broker_status[\"funds_endpoint_verified\"]:\n        missing.append(\"funds_endpoint_unverified\")\n''',
    label="live order readiness missing reason",
)
text = replace_once(
    text,
    '''    reconciliation_completed = bool(\n        getattr(ctx, \"position_reconciliation_completed\", False)\n    ) and bool(structured_status.get(\"order_endpoint_verified\", False))\n''',
    '''    reconciliation_completed = bool(\n        getattr(ctx, \"position_reconciliation_completed\", False)\n    )\n''',
    label="decouple reconciliation truth from order endpoint telemetry",
)
write(path, text)


# ---------------------------------------------------------------------------
# 3) Quote freshness authority: only the MDM live-receipt timestamp outranks
#    mutable cached age fields. Generic historical/replay timestamp_ms remains
#    backward-compatible and is intentionally not treated as live proof.
# ---------------------------------------------------------------------------
path = "src/nifty_scalper_bot/execution/quote_readiness.py"
text = read(path)
text = replace_once(
    text,
    '''from dataclasses import asdict, dataclass\nfrom typing import Any, Mapping\n''',
    '''import time\nfrom dataclasses import asdict, dataclass\nfrom typing import Any, Mapping\n''',
    label="quote readiness time import",
)
text = replace_once(
    text,
    '''    quote_timestamp_quality_allows_hard_readiness,\n    resolve_quote_bid_ask_spread,\n    resolve_quote_age_seconds,\n''',
    '''    quote_timestamp_quality_allows_hard_readiness,\n    resolve_quote_age_seconds,\n    resolve_quote_bid_ask_spread,\n''',
    label="quote readiness import ordering",
)
text = replace_once(
    text,
    '''def resolve_tick_age_ms(payload: Mapping[str, Any] | object | None) -> float | None:\n    age_s = resolve_quote_age_seconds(payload)\n    return None if age_s is None else age_s * 1000.0\n''',
    '''def resolve_tick_age_ms(payload: Mapping[str, Any] | object | None) -> float | None:\n    timestamp_ms = _float(payload, \"last_tick_ts_ms\")\n    if timestamp_ms is not None and timestamp_ms > 10_000_000_000:\n        return max(0.0, time.time() * 1000.0 - timestamp_ms)\n    age_s = resolve_quote_age_seconds(payload)\n    return None if age_s is None else age_s * 1000.0\n''',
    label="live receipt timestamp precedence over cached quote age",
)
write(path, text)
