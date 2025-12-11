"""Deterministic .env loader executed before any config imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    ROOT = Path(__file__).resolve().parent.parent.parent
    ENV_FILE = ROOT / ".env"
    ENV_LOCAL = ROOT / ".env.local"

    print(f"🔍 [ENV-LOADER] ROOT: {ROOT}", file=sys.stderr)
    print(f"🔍 [ENV-LOADER] .env exists: {ENV_FILE.exists()}", file=sys.stderr)
    print(f"🔍 [ENV-LOADER] .env path: {ENV_FILE}", file=sys.stderr)

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
        print(f"✅ [ENV-LOADER] Loaded .env from {ENV_FILE}", file=sys.stderr)
        # Verify critical variables
        log_level = os.getenv("LOG_LEVEL", "NOT_SET")
        elite_enabled = os.getenv("ELITE_STRATEGIES_ENABLED", "NOT_SET")
        print(f"✅ [ENV-LOADER] LOG_LEVEL={log_level}", file=sys.stderr)
        print(f"✅ [ENV-LOADER] ELITE_STRATEGIES_ENABLED={elite_enabled}", file=sys.stderr)
    else:
        print(f"⚠️ [ENV-LOADER] .env file NOT FOUND at {ENV_FILE}", file=sys.stderr)
        
    if ENV_LOCAL.exists():
        load_dotenv(ENV_LOCAL, override=True)
        print(f"✅ [ENV-LOADER] Loaded .env.local from {ENV_LOCAL}", file=sys.stderr)
except ImportError:  # pragma: no cover - optional dependency
    print("Warning: python-dotenv not installed", file=sys.stderr)

try:
    from nifty_scalper_bot.config.env_aliases import normalize_env_on_load

    normalize_env_on_load()
    print("✅ [ENV-LOADER] Environment aliases normalized", file=sys.stderr)
except ImportError:  # pragma: no cover - bootstrap fallback
    pass
except Exception as exc:  # noqa: BLE001
    print(
        f"Warning: Failed to normalize environment aliases: {exc}",
        file=sys.stderr,
    )
