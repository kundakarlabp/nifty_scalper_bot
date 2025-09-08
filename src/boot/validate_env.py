from __future__ import annotations

"""Runtime environment validation and setup helpers."""

from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from typing import Optional, cast
from decimal import Decimal

import pandas as pd
import yaml  # type: ignore[import-untyped]

from src.config import AppSettings, settings
from src.utils.market_time import IST, is_market_open, prev_session_last_20m

try:  # Optional broker SDK
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore


ENABLE_LIVE_TRADING = str(
    os.getenv("ENABLE_LIVE_TRADING")
    or os.getenv("ENABLE_TRADING")
    or "false"
).lower() in {"1", "true", "yes"}

SKIP_BROKER_VALIDATION = str(
    os.getenv("SKIP_BROKER_VALIDATION", "false")
).lower() in {"1", "true", "yes"}


def env_any(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable from ``names``."""
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


API_KEY_ALIASES = ("ZERODHA__API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
API_SECRET_ALIASES = ("ZERODHA__API_SECRET", "ZERODHA_API_SECRET", "KITE_API_SECRET")
ACCESS_TOKEN_ALIASES = (
    "ZERODHA__ACCESS_TOKEN",
    "ZERODHA_ACCESS_TOKEN",
    "KITE_ACCESS_TOKEN",
)


def _read_creds() -> dict[str, str | None]:
    """Read Zerodha credentials from any supported alias."""
    return {
        "api_key": env_any(*API_KEY_ALIASES),
        "api_secret": env_any(*API_SECRET_ALIASES),
        "access_token": env_any(*ACCESS_TOKEN_ALIASES),
    }


def _export_aliases_into_environ() -> None:
    """Populate missing alias env vars with the available credential values."""
    creds = _read_creds()
    alias_map = {
        "api_key": API_KEY_ALIASES,
        "api_secret": API_SECRET_ALIASES,
        "access_token": ACCESS_TOKEN_ALIASES,
    }
    for key, aliases in alias_map.items():
        val = creds.get(key)
        if val:
            for name in aliases:
                os.environ.setdefault(name, val)


def _skip_validation() -> bool:
    return SKIP_BROKER_VALIDATION


API_KEY = env_any(*API_KEY_ALIASES)
API_SECRET = env_any(*API_SECRET_ALIASES)
ACCESS_TOKEN = env_any(*ACCESS_TOKEN_ALIASES)

# Deployment environment flags
# True when running on Railway (detected via known env vars)
IS_HOSTED_RAILWAY = bool(env_any("RAILWAY_PROJECT_ID", "RAILWAY_STATIC_URL"))

# Optional runtime flags
BROKER_CONNECT_FOR_DATA = (
    str(os.getenv("BROKER_CONNECT_FOR_DATA", "false")).lower()
    in {"1", "true", "yes"}
)
DATA_WARMUP_DISABLE = (
    str(
        os.getenv(
            "DATA__WARMUP_DISABLE",
            "true" if IS_HOSTED_RAILWAY else "false",
        )
    ).lower()
    in {"1", "true", "yes"}
)


def seed_env_from_defaults(path: str = "config/defaults.yaml") -> None:
    """Populate ``os.environ`` with values from a defaults YAML file.

    Existing environment variables take precedence and are not overridden.
    Nested keys in the YAML are flattened using ``__`` to mirror Pydantic's
    ``env_nested_delimiter`` behaviour.
    """

    p = Path(path)
    if not p.is_file():
        return

    try:
        data = yaml.safe_load(p.read_text("utf-8")) or {}
    except Exception:  # pragma: no cover - defensive
        logging.getLogger("config").exception("Failed loading %s", p)
        return

    def _flatten(prefix: str, obj: dict[str, object]) -> None:
        for k, v in obj.items():
            key = f"{prefix}{k}".upper()
            if isinstance(v, dict):
                _flatten(f"{key}__", v)
            else:
                os.environ.setdefault(key, str(v))

    _flatten("", data)


def validate_critical_settings(cfg: Optional[AppSettings] = None) -> None:
    """Perform runtime checks on essential configuration values."""

    _export_aliases_into_environ()
    cfg = cast(AppSettings, cfg or settings)
    _log_cred_presence()
    errors: list[str] = []

    if _skip_validation():
        logging.getLogger(__name__).warning(
            "SKIP_BROKER_VALIDATION=true: proceeding without live creds",
        )
        return

    # Live trading requires broker credentials
    if cfg.enable_live_trading:
        if not cfg.zerodha.api_key:
            errors.append(
                "/".join(API_KEY_ALIASES)
                + " is required when ENABLE_LIVE_TRADING=true",
            )
        if not cfg.zerodha.api_secret:
            errors.append(
                "/".join(API_SECRET_ALIASES)
                + " is required when ENABLE_LIVE_TRADING=true",
            )
        if not cfg.zerodha.access_token:
            errors.append(
                "/".join(ACCESS_TOKEN_ALIASES)
                + " is required when ENABLE_LIVE_TRADING=true",
            )

    # Telegram configuration (required only when enabled)
    if cfg.telegram.enabled:
        if not cfg.telegram.bot_token:
            errors.append("TELEGRAM__BOT_TOKEN is required when TELEGRAM__ENABLED=true")
        if not cfg.telegram.chat_id:
            errors.append("TELEGRAM__CHAT_ID is required when TELEGRAM__ENABLED=true")

    # Ensure lookback window can satisfy minimum bars requirement
    if cfg.data.lookback_minutes < cfg.strategy.min_bars_for_signal:
        errors.append(
            "DATA__LOOKBACK_MINUTES must be >= STRATEGY__MIN_BARS_FOR_SIGNAL",
        )

    # Instrument token sanity check (only in live mode with deps available)
    if (
        cfg.enable_live_trading
        and not cfg.allow_offhours_testing
        and KiteConnect is not None
        and LiveKiteSource is not None
        and not errors  # only attempt if creds are present
    ):
        token = int(getattr(cfg.instruments, "instrument_token", 0) or 0)
        if token <= 0:
            errors.append("INSTRUMENTS__INSTRUMENT_TOKEN must be a positive integer")
        else:
            src = None
            try:
                kite = KiteConnect(api_key=str(cfg.zerodha.api_key))
                kite.set_access_token(str(cfg.zerodha.access_token))
                src = LiveKiteSource(kite=kite)
                try:
                    src.connect()
                    now_ist = datetime.now(IST).replace(
                        second=0, microsecond=0,
                    )
                    if is_market_open(now_ist):
                        start = now_ist - timedelta(minutes=20)
                        end = now_ist
                    else:
                        start, end = prev_session_last_20m(now_ist)
                    df = src.fetch_ohlc(
                        token=token, start=start, end=end, timeframe="minute",
                    )
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        # Outside market hours the historical API may return no data.
                        # Fall back to a simple last-price lookup so that a valid token
                        # doesn't trigger a false validation error.
                        ltp_fn = getattr(src, "get_last_price", None)
                        if not callable(ltp_fn):
                            errors.append(
                                f"instrument_token {token} returned no data; configure a valid F&O token",
                            )
                        else:  # pragma: no cover - network dependent
                            try:
                                ltp = ltp_fn(token)
                            except Exception:
                                ltp = None
                            if not isinstance(ltp, (int, float)):
                                logging.getLogger("config").warning(
                                    "instrument_token %s returned no data", token
                                )
                finally:
                    disconnect = getattr(src, "disconnect", None)
                    if callable(disconnect):
                        try:
                            disconnect()
                        except Exception:
                            logging.getLogger("config").debug(
                                "LiveKiteSource disconnect failed", exc_info=True,
                            )
            except Exception as e:
                logging.getLogger("config").warning(
                    "instrument_token validation skipped: %s", e,
                )
            finally:
                if src is not None:
                    disconnect_fn = getattr(src, "disconnect", None)
                    if callable(disconnect_fn):
                        try:
                            disconnect_fn()
                        except Exception:
                            logging.getLogger("config").debug(
                                "instrument_token validation disconnect failed", exc_info=True,
                            )

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))


def validate_runtime_env(cfg: Optional[AppSettings] = None) -> None:
    """Light-weight runtime checks for critical dependencies.

    This verifies that mandatory environment variables are present, the
    instruments CSV exists, and that the broker SDK can be instantiated with the
    provided tokens.  The checks intentionally avoid any network calls so they
    remain fast and side‑effect free.
    """

    cfg = cast(AppSettings, cfg or settings)
    errors: list[str] = []

    # --- required environment variables ---
    required = [
        API_KEY_ALIASES,
        API_SECRET_ALIASES,
        ACCESS_TOKEN_ALIASES,
    ]
    if cfg.enable_live_trading and not _skip_validation():
        for keys in required:
            if not env_any(*keys):
                errors.append("/".join(keys) + " must be set in the environment")

    # --- instrument CSV path ---
    csv_path = Path(os.environ.get("INSTRUMENTS_CSV", "data/nifty_ohlc.csv"))
    if not csv_path.exists():
        errors.append(f"Instrument CSV not found: {csv_path}")

    # --- broker connectivity & tokens ---
    if cfg.enable_live_trading and KiteConnect is not None and not _skip_validation():
        try:
            kite = KiteConnect(api_key=str(cfg.zerodha.api_key))
            kite.set_access_token(str(cfg.zerodha.access_token))
            # avoid network calls; simply ensure object creation succeeded
            _ = Decimal("0")  # exercise Decimal import for mypy
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Broker SDK initialisation failed: {exc}")

    if errors:
        if _skip_validation():
            logging.getLogger(__name__).warning(
                "SKIP_BROKER_VALIDATION=true: proceeding without live creds",
            )
            return
        raise RuntimeError("Runtime validation failed:\n" + "\n".join(errors))


def _log_cred_presence() -> None:
    """Log the presence of live trading credentials without revealing them."""
    log = logging.getLogger(__name__)
    creds = _read_creds()

    def mask(v: str | None) -> bool:
        return bool(v and v.strip())

    log.info(
        "live=%s (env=%s), skip_validation=%s, api_key=%s, secret=%s, access=%s",
        settings.enable_live_trading,
        ENABLE_LIVE_TRADING,
        SKIP_BROKER_VALIDATION,
        mask(creds["api_key"]),
        mask(creds["api_secret"]),
        mask(creds["access_token"]),
    )

