"""
Centralized data path configuration.

All file persistence should use get_data_path() to ensure
proper permissions in containerized environments.
"""

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Return writable data directory with safe fallback."""
    candidates = [
        os.getenv("DATA_DIR"),
        "/app/data",
        str(Path.cwd() / "data"),
        "/tmp/nifty_scalper_bot_data",
    ]

    last_error: Exception | None = None
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return p
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"No writable data directory found: {last_error}")


def get_data_path(filename: str) -> Path:
    """Get full path for a data file.
    
    Usage:
        from nifty_scalper_bot.config.paths import get_data_path
        
        trades_file = get_data_path("trades.json")
        orders_file = get_data_path("order_history.json")
    """
    return get_data_dir() / filename


# For backward compatibility
TRADES_FILE = get_data_path("trades.json")
ORDER_HISTORY_FILE = get_data_path("order_history.json")
BRACKETS_FILE = get_data_path("brackets.json")
