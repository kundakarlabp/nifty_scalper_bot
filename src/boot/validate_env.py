import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, cast

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


def env_any(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable from ``names``."""
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


def _skip_validation() -> bool:
    return skip_broker_validation()


# Exposed flag so tests can monkeypatch and skip broker checks
SKIP_BROKER_VALIDATION: bool = str(
    os.getenv("SKIP_BROKER_VALIDATION", "false")
).lower() in {"1", "true", "yes"}


KITE_API_KEY_NAMES: tuple[str, ...] = ("KITE_API_KEY",)
KITE_API_SECRET_NAMES: tuple[str, ...] = ("KITE_API_SECRET",)
KITE_ACCESS_TOKEN_NAMES: tuple[str, ...] = ("KITE_ACCESS_TOKEN",)
INSTRUMENTS_CSV_ALIASES: tuple[str, ...] = (
    "INSTRUMENTS__CSV",
    "INSTRUMENTS_CSV",
)

API_KEY = env_any(*KITE_API_KEY_NAMES)
API_SECRET = env_any(*KITE_API_SECRET_NAMES)
ACCESS_TOKEN = env_any(*KITE_ACCESS_TOKEN_NAMES)

# Deployment environment flags
# True when running on Railway (detected via known env vars)
IS_HOSTED_RAILWAY = bool(env_any("RAILWAY_PROJECT_ID", "RAILWAY_STATIC_URL"))

_FALLBACK_WARNINGS_EMITTED: set[str] = set()
_ENV_PROBE_LOGGED = False


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


_ENV_SEEDED = False


def _ensure_env_seeded() -> None:
    global _ENV_SEEDED
    if not _ENV_SEEDED:
        seed_env_from_defaults()
        _ENV_SEEDED = True


def enable_live_trading() -> bool:
    _ensure_env_seeded()
    return str(
        os.getenv("ENABLE_LIVE_TRADING") or os.getenv("ENABLE_TRADING") or "false"
    ).lower() in {"1", "true", "yes"}


def skip_broker_validation() -> bool:
    """Return True when broker validation should be skipped.

    Respects the module-level ``SKIP_BROKER_VALIDATION`` flag so tests can
    monkeypatch it without relying on environment variables.
    """
    _ensure_env_seeded()
    if SKIP_BROKER_VALIDATION:
        return True
    return str(os.getenv("SKIP_BROKER_VALIDATION", "false")).lower() in {
        "1",
        "true",
        "yes",
    }


def broker_connect_for_data() -> bool:
    _ensure_env_seeded()
    return str(os.getenv("BROKER_CONNECT_FOR_DATA", "false")).lower() in {
        "1",
        "true",
        "yes",
    }


def data_warmup_disable() -> bool:
    _ensure_env_seeded()
    val = os.getenv("DATA__WARMUP_DISABLE")
    if val is None:
        val = os.getenv("DATA_WARMUP_DISABLE", "false")
    return str(val).lower() in {"1", "true", "yes"}


def data_warmup_backfill_min() -> int:
    _ensure_env_seeded()
    val = os.getenv("DATA__WARMUP_BACKFILL_MIN")
    if val is None:
        val = os.getenv("DATA_WARMUP_BACKFILL_MIN", "15")
    try:
        return max(0, int(val))
    except Exception:
        return 15


def data_allow_synthetic_on_empty() -> bool:
    _ensure_env_seeded()
    val = os.getenv("DATA__ALLOW_SYNTHETIC_ON_EMPTY")
    if val is None:
        val = os.getenv("DATA_ALLOW_SYNTHETIC_ON_EMPTY", "true")
    return str(val).lower() in {"1", "true", "yes"}


def data_clamp_to_market_open() -> bool:
    _ensure_env_seeded()
    val = os.getenv("DATA__CLAMP_TO_MARKET_OPEN")
    if val is None:
        val = os.getenv("DATA_CLAMP_TO_MARKET_OPEN", "false")
    return str(val).lower() in {"1", "true", "yes"}


def validate_critical_settings(cfg: Optional[AppSettings] = None) -> None:
    """Perform runtime checks on essential configuration values."""

    materialized = AppSettings()  # type: ignore[call-arg]
    if cfg is None:
        loader = getattr(settings, "_load", None)
        if callable(loader):
            cfg = cast(AppSettings, loader())
        else:  # pragma: no cover - defensive fallback
            cfg = cast(AppSettings, settings)
    cfg = cast(AppSettings, cfg or materialized)

    def _is_present(val: object) -> bool:
        if isinstance(val, bool):
            return bool(val)
        return bool(val and str(val).strip())

    def _resolve_primary(primary: str) -> object | None:
        value = getattr(cfg, primary, None)
        if _is_present(value):
            return value
        attr = {
            "KITE_API_KEY": "api_key",
            "KITE_API_SECRET": "api_secret",
            "KITE_ACCESS_TOKEN": "access_token",
        }.get(primary)
        if attr is None:
            return None
        kite_cfg = getattr(cfg, "kite", None)
        nested = getattr(kite_cfg, attr, None)
        return nested if _is_present(nested) else None

    def _warn_fallback_once(primary: str, fallback: str) -> None:
        if primary in _FALLBACK_WARNINGS_EMITTED:
            return
        logging.getLogger(__name__).warning(
            "Using %s as fallback for missing %s", fallback, primary
        )
        _FALLBACK_WARNINGS_EMITTED.add(primary)

    def _resolve_credential(primary: str, fallback: str) -> tuple[object | None, bool]:
        value = _resolve_primary(primary)
        if _is_present(value):
            return value, False
        fallback_val = getattr(cfg, fallback, None)
        if _is_present(fallback_val):
            _warn_fallback_once(primary, fallback)
            return fallback_val, True
        return None, False

    live = bool(cfg.enable_live_trading)
    live_flag = getattr(cfg, "ENABLE_LIVE_TRADING", None)
    fields_set: set[str] = getattr(cfg, "model_fields_set", set())
    if isinstance(live_flag, bool) and "ENABLE_LIVE_TRADING" in fields_set:
        live = live_flag

    key_value, _ = _resolve_credential("KITE_API_KEY", "ZERODHA_API_KEY")
    secret_value, _ = _resolve_credential("KITE_API_SECRET", "ZERODHA_API_SECRET")
    token_value, _ = _resolve_credential("KITE_ACCESS_TOKEN", "ZERODHA_ACCESS_TOKEN")

    have_key = _is_present(key_value)
    have_secret = _is_present(secret_value)
    have_token = _is_present(token_value)

    resolved_api_key = str(key_value).strip() if have_key else ""
    resolved_access_token = str(token_value).strip() if have_token else ""

    global _ENV_PROBE_LOGGED
    if not _ENV_PROBE_LOGGED:
        logging.getLogger(__name__).info(
            "env_probe live=%s have_key=%s have_secret=%s have_access=%s",
            live,
            have_key,
            have_secret,
            have_token,
            extra={
                "live": live,
                "have_key": have_key,
                "have_secret": have_secret,
                "have_access": have_token,
            },
        )
        _ENV_PROBE_LOGGED = True

    if _skip_validation():
        logging.getLogger(__name__).warning(
            "SKIP_BROKER_VALIDATION=true: proceeding without live creds",
        )
        return

    if live:
        missing = []
        if not have_key:
            missing.append("KITE_API_KEY")
        if not have_secret:
            missing.append("KITE_API_SECRET")
        if not have_token:
            missing.append("KITE_ACCESS_TOKEN")
        if missing:
            missing_csv = ", ".join(missing)
            raise ValueError(
                "ENABLE_LIVE_TRADING=true requires KITE_API_KEY, KITE_API_SECRET, "
                "and the daily KITE_ACCESS_TOKEN; missing: "
                f"{missing_csv}"
            )

    errors: list[str] = []

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
        live
        and not cfg.allow_offhours_testing
        and KiteConnect is not None
        and LiveKiteSource is not None
        and not errors  # only attempt if creds are present
    ):
        token = int(getattr(cfg.instruments, "instrument_token", 0) or 0)
        src = None
        try:
            kite = KiteConnect(api_key=resolved_api_key or str(cfg.kite.api_key))
            kite.set_access_token(
                resolved_access_token or str(cfg.kite.access_token)
            )
            src = LiveKiteSource(kite=kite)
            try:
                src.connect()
                now_ist = datetime.now(IST).replace(
                    second=0,
                    microsecond=0,
                )
                if is_market_open(now_ist):
                    start = now_ist - timedelta(minutes=20)
                    end = now_ist
                else:
                    start, end = prev_session_last_20m(now_ist)
                res = src.fetch_ohlc(
                    token=token,
                    start=start,
                    end=end,
                    timeframe="minute",
                )
                df = res.df
                if df.empty:
                    # Outside market hours the historical API may return no data.
                    # Fall back to a simple last-price lookup so that a valid token
                    # doesn't trigger a false validation error.
                    ltp_fn = getattr(src, "get_last_price", None)
                    ltp = ltp_fn(token) if callable(ltp_fn) else None
                    if not isinstance(ltp, (int, float)):
                        errors.append(
                            f"instrument_token {token} returned no data; configure a valid F&O token",
                        )
            finally:
                disconnect = getattr(src, "disconnect", None)
                if callable(disconnect):
                    try:
                        disconnect()
                    except Exception:
                        logging.getLogger("config").debug(
                            "LiveKiteSource disconnect failed",
                            exc_info=True,
                        )
        except Exception as e:
            logging.getLogger("config").warning(
                "instrument_token validation skipped: %s",
                e,
            )
        finally:
            if src is not None:
                disconnect_fn = getattr(src, "disconnect", None)
                if callable(disconnect_fn):
                    try:
                        disconnect_fn()
                    except Exception:
                        logging.getLogger("config").debug(
                            "instrument_token validation disconnect failed",
                            exc_info=True,
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
        KITE_API_KEY_NAMES,
        KITE_API_SECRET_NAMES,
        KITE_ACCESS_TOKEN_NAMES,
    ]
    if cfg.enable_live_trading and not _skip_validation():
        for keys in required:
            if not env_any(*keys):
                errors.append("/".join(keys) + " must be set in the environment")

    # --- instrument CSV path ---
    csv_env = env_any(*INSTRUMENTS_CSV_ALIASES)
    csv_path_value = getattr(cfg, "INSTRUMENTS_CSV", "") or csv_env
    csv_path = Path(csv_path_value or "data/nifty_ohlc.csv")
    if not csv_path.exists():
        errors.append(f"Instrument CSV not found: {csv_path}")

    # --- broker connectivity & tokens ---
    if cfg.enable_live_trading and KiteConnect is not None and not _skip_validation():
        try:
            kite = KiteConnect(api_key=str(cfg.kite.api_key))
            kite.set_access_token(str(cfg.kite.access_token))
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

    def mask(v: str | None) -> bool:
        return bool(v and v.strip())

    log.info(
        "live=%s (env=%s), skip_validation=%s, api_key=%s, secret=%s, access=%s",
        settings.enable_live_trading,
        enable_live_trading(),
        skip_broker_validation(),
        mask(API_KEY),
        mask(API_SECRET),
        mask(ACCESS_TOKEN),
    )
