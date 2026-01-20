"""
Centralized data path configuration.

All file persistence should use get_data_path() to ensure
proper permissions in containerized environments.
"""

import os
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    """Get the writable data directory.
    
    Priority:
    1. DATA_DIR environment variable
    2. /app/data (if exists and writable)
    3. ./data (current working directory)
    4. /tmp/nifty_scalper_data (always writable fallback)
    """
    # Check DATA_DIR env var first
    data_dir_env = os.getenv("DATA_DIR")
    if data_dir_env:
        path = Path(data_dir_env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # Try /app/data
    app_data = Path("/app/data")
    if app_data.exists():
        try:
            test_file = app_data / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return app_data
        except (PermissionError, OSError):
            pass
    
    # Try ./data (relative)
    cwd_data = Path.cwd() / "data"
    try:
        cwd_data.mkdir(parents=True, exist_ok=True)
        test_file = cwd_data / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return cwd_data
    except (PermissionError, OSError):
        pass
    
    # Fallback to /tmp
    tmp_data = Path("/tmp/nifty_scalper_data")
    tmp_data.mkdir(parents=True, exist_ok=True)
    return tmp_data


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
