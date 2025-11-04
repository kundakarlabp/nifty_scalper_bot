"""Deterministic .env loader executed before any config imports."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    ROOT = Path(__file__).resolve().parent.parent.parent
    ENV_FILE = ROOT / ".env"
    ENV_LOCAL = ROOT / ".env.local"

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    if ENV_LOCAL.exists():
        load_dotenv(ENV_LOCAL, override=True)
except ImportError:  # pragma: no cover - optional dependency
    print("Warning: python-dotenv not installed", file=sys.stderr)

try:
    from nifty_scalper_bot.config.env_aliases import normalize_env_on_load

    normalize_env_on_load()
except ImportError:  # pragma: no cover - bootstrap fallback
    pass
except Exception as exc:  # noqa: BLE001
    print(
        f"Warning: Failed to normalize environment aliases: {exc}",
        file=sys.stderr,
    )
