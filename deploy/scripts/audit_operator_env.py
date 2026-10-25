"""Purpose: Audit protected operator env configuration without changing it.

Responsibilities:
- Read only one explicitly named protected file, never a fallback dotenv.
- Report recognized GitHub-owned policy overrides and duplicate mode settings.
- Emit fixed classifications and allowlisted key names, never raw values.

Boundaries: Operator-only diagnostic; not part of runtime or deployment gating.
Safety: No broker/network calls, credential output, imports or file mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path

_MAX_SIZE = 256 * 1024
_KEY_LINE = re.compile(r"^(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)$")
_STRATEGY_RISK_KEYS = frozenset(
    {
        "RISK__PER_TRADE_RISK_PCT",
        "RISK_PER_TRADE_PCT",
        "RISK_DAILY_LOSS_PCT",
        "RISK_DAILY_PNL_CAP_PCT",
        "RISK_MAX_DAILY_LOSS_PCT",
        "DAILY_PNL_CAP_PCT",
        "SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED",
        "VWAP_TREND_RETEST_EXTENSION_ENABLED",
        "VWAP_TREND_RETEST_EXTENSION_MAX_ATR",
        "ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED",
        "MIN_SIGNAL_SCORE",
        "CONFIDENCE_THRESHOLD",
        "MAX_TRADES_PER_DAY",
        "MAX_TRADES_DAY",
        "ATR_SL_MULTIPLIER",
        "ATR_TP_MULTIPLIER",
    }
)
_MONITORING_KEYS = frozenset(
    {
        "SUPABASE_RUNTIME_HEARTBEAT_ENABLED",
        "SUPABASE_RUNTIME_HEARTBEAT_URL",
        "SUPABASE_RUNTIME_HEARTBEAT_INTERVAL_SECONDS",
        "SUPABASE_LOG_ARCHIVE_ENABLED",
        "SUPABASE_LOG_ARCHIVE_URL",
        "SUPABASE_LOG_ARCHIVE_INTERVAL_SECONDS",
        "SUPABASE_TRADE_REPLICATION_ENABLED",
        "SUPABASE_TRADE_REPLICATION_URL",
        "SUPABASE_TRADE_REPLICATION_INTERVAL_SECONDS",
    }
)
_MODE_KEYS = frozenset({"ENABLE_LIVE", "EXECUTION_MODE"})
_HOST_KEYS = frozenset(
    {
        "GIT_COMMIT_SHA",
        "PORT",
        "BOT_APP_DIR",
        "BOT_ENV_FILE",
        "BOT_SERVICE_NAME",
        "BOT_ADMIN_PORT",
        "BOT_STREAMLIT_PORT",
        "BOT_RESEARCH_LAUNCHER",
        "DEPLOYMENT_PLATFORM",
        "PRODUCTION_LIVE_DEFAULT_INITIALIZED",
        "PAPER_MODE",
        "SHADOW_MODE",
        "ENABLE_LIVE_TRADING",
        "ORDERS__ENABLE_LIVE",
        "PAPER__ENABLED",
    }
)
_SECRET_KEY = re.compile(
    r"^(?:KITE_|ZERODHA_|BROKER_|TELEGRAM_|ADMIN_|SUPABASE_(?:SECRET|SERVICE|ANON|KEY))"
    r"|(?:TOKEN|SECRET|PASSWORD|PASSPHRASE|PRIVATE_KEY|API_KEY)$"
)


def _read_protected(path: Path) -> str | None:
    """Open one private regular file without following symlinks or printing paths."""
    if not path.is_absolute():
        return None
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o077
            or info.st_size > _MAX_SIZE
        ):
            return None
        data = os.read(descriptor, _MAX_SIZE + 1)
        if len(data) > _MAX_SIZE:
            return None
        return data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _mode_alignment(values: dict[str, str], duplicates: set[str]) -> str:
    if duplicates & _MODE_KEYS:
        return "ambiguous"
    live = values.get("ENABLE_LIVE", "").strip().lower()
    mode = values.get("EXECUTION_MODE", "").strip().upper()
    if not live or not mode:
        return "incomplete"
    if live in {"true", "1", "yes", "on"} and mode == "LIVE":
        return "explicit_live"
    if live in {"false", "0", "no", "off"} and mode in {"SHADOW", "PAPER"}:
        return "explicit_nonlive"
    return "conflict"


def audit(path: Path) -> dict[str, object]:
    """Classify host overrides; no raw source value appears in the result."""
    raw = _read_protected(path)
    if raw is None:
        return {"status": "missing_or_unsafe_file"}
    values: dict[str, str] = {}
    counts: Counter[str] = Counter()
    unknown = 0
    malformed = 0
    known = _STRATEGY_RISK_KEYS | _MONITORING_KEYS | _MODE_KEYS | _HOST_KEYS
    for line in raw.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        match = _KEY_LINE.fullmatch(clean)
        if match is None:
            malformed += 1
            continue
        name, value = match.groups()
        counts[name] += 1
        # Preserve only the two boolean/mode values for fixed classification.
        if name in _MODE_KEYS:
            values[name] = value.strip().strip('"').strip("'")
        if name not in known and not _SECRET_KEY.search(name):
            unknown += 1
    duplicate = {name for name, n in counts.items() if n > 1}
    return {
        "status": "ok",
        "mode_alignment": _mode_alignment(values, duplicate),
        "strategy_risk_overrides": sorted(_STRATEGY_RISK_KEYS & counts.keys()),
        "monitoring_overrides": sorted(_MONITORING_KEYS & counts.keys()),
        "duplicate_policy_keys": sorted(duplicate & known),
        "unclassified_key_count": unknown,
        "malformed_line_count": malformed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only protected env drift audit")
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.env_file)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
