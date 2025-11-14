"""Environment configuration loader."""

from __future__ import annotations

import os
from typing import Literal, cast

from pydantic import HttpUrl, TypeAdapter
from pydantic_core import Url

from nifty_scalper_bot.config import defaults
from nifty_scalper_bot.config.base import (
    AppConfig,
    BrokerConfig,
    LoggingConfig,
    RateLimitBucketConfig,
    RateLimitConfig,
    RiskConfig,
    TelegramConfig,
)
from nifty_scalper_bot.utils.env import get_bool, get_csv, get_float, get_int
from nifty_scalper_bot.utils.errors import ConfigurationError

URL_ADAPTER = TypeAdapter(Url)
HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


def _first_env(*names: str, default: str | None = None) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    if default is not None:
        return default
    joined = ", ".join(names)
    raise ConfigurationError(f"Missing required env: any of {joined}")


def _optional_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _get_optional_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        int(str(value).strip())
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer for {name}: {value}") from exc
    return get_int(name, default)


def _get_optional_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        float(str(value).strip())
    except ValueError as exc:
        raise ConfigurationError(f"Invalid float for {name}: {value}") from exc
    return get_float(name, default)


def _get_optional_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return get_bool(name, default)
    if lowered in {"0", "false", "no", "off"}:
        return get_bool(name, default)
    raise ConfigurationError(f"Invalid boolean for {name}: {value}")




def POLL_INTERVAL_SECS_MARKET() -> float:
    """Return polling interval in seconds when markets are open."""

    return _get_optional_float('POLL_INTERVAL_SECS_MARKET', 0.5)


def POLL_INTERVAL_SECS_OFFHOURS() -> float:
    """Return polling interval in seconds for off-hours cadence."""

    return _get_optional_float('POLL_INTERVAL_SECS_OFFHOURS', 5.0)


def POLL_RECONNECT_CAP_SECS() -> int:
    """Return the cap for polling reconnect backoff seconds."""

    return _get_optional_int('POLL_RECONNECT_CAP_SECS', 30)


def _get_cred(var_primary: str, *aliases: str) -> str:
    names = (var_primary,) + aliases
    for name in names:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    raise ConfigurationError(f"Missing required credential: any of {names}")


def build_broker_config() -> BrokerConfig:
    api_key = _get_cred("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
    api_secret = _get_cred("BROKER_API_SECRET", "ZERODHA_API_SECRET", "KITE_API_SECRET")
    base_url_value = (
        os.environ.get("BROKER_BASE_URL")
        or os.environ.get("ZERODHA_BASE_URL")
        or defaults.DEFAULT_BROKER_BASE_URL
    )
    base_url = HTTP_URL_ADAPTER.validate_python(base_url_value)
    ws_default = getattr(
        defaults, "DEFAULT_BROKER_WS_URL", defaults.DEFAULT_BROKER_WEBSOCKET_URL
    )
    ws_url = (
        os.environ.get("BROKER_WS_URL")
        or os.environ.get("ZERODHA_WS_URL")
        or ws_default
    )
    access_token = (
        os.environ.get("BROKER_ACCESS_TOKEN")
        or os.environ.get("ZERODHA_ACCESS_TOKEN")
        or os.environ.get("KITE_ACCESS_TOKEN")
        or ""
    )
    return BrokerConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        websocket_url=ws_url,
        access_token=access_token,
    )


def load_from_env() -> AppConfig:
    """Construct an :class:`AppConfig` from environment variables."""

    base_url_value = _first_env(
        "BROKER_BASE_URL", "ZERODHA_BASE_URL", default=defaults.DEFAULT_BROKER_BASE_URL
    )
    base_url = HTTP_URL_ADAPTER.validate_python(base_url_value)
    broker = BrokerConfig(
        api_key=_first_env("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY"),
        api_secret=_first_env(
            "BROKER_API_SECRET", "ZERODHA_API_SECRET", "KITE_API_SECRET"
        ),
        access_token=(
            os.environ.get("BROKER_ACCESS_TOKEN")
            or os.environ.get("ZERODHA_ACCESS_TOKEN")
            or os.environ.get("KITE_ACCESS_TOKEN")
        ),
        base_url=base_url,
        websocket_url=_first_env(
            "BROKER_WS_URL",
            "ZERODHA_WS_URL",
            default=defaults.DEFAULT_BROKER_WEBSOCKET_URL,
        ),
    )

    risk = RiskConfig(
        max_daily_trades=_get_optional_int(
            "RISK_MAX_DAILY_TRADES", defaults.DEFAULT_RISK_MAX_DAILY_TRADES
        ),
        max_order_notional=_get_optional_float(
            "RISK_MAX_ORDER_NOTIONAL", defaults.DEFAULT_RISK_MAX_ORDER_NOTIONAL
        ),
        allow_short=_get_optional_bool(
            "RISK_ALLOW_SHORT", defaults.DEFAULT_RISK_ALLOW_SHORT
        ),
        max_drawdown_pct=_get_optional_float(
            "RISK_MAX_DRAWDOWN_PCT", defaults.DEFAULT_RISK_MAX_DRAWDOWN_PCT
        ),
        max_daily_loss_pct=_get_optional_float(
            "RISK_MAX_DAILY_LOSS_PCT", defaults.DEFAULT_RISK_MAX_DAILY_LOSS_PCT
        ),
        max_position_size_pct=_get_optional_float(
            "RISK_MAX_POSITION_SIZE_PCT",
            defaults.DEFAULT_RISK_MAX_POSITION_SIZE_PCT,
        ),
        max_total_exposure_pct=_get_optional_float(
            "RISK_MAX_TOTAL_EXPOSURE_PCT",
            defaults.DEFAULT_RISK_MAX_TOTAL_EXPOSURE_PCT,
        ),
        max_concurrent_positions=_get_optional_int(
            "RISK_MAX_CONCURRENT_POSITIONS",
            defaults.DEFAULT_RISK_MAX_CONCURRENT_POSITIONS,
        ),
        max_positions_per_symbol=_get_optional_int(
            "RISK_MAX_POSITIONS_PER_SYMBOL",
            defaults.DEFAULT_RISK_MAX_POSITIONS_PER_SYMBOL,
        ),
        min_risk_reward_ratio=_get_optional_float(
            "RISK_MIN_RISK_REWARD_RATIO",
            defaults.DEFAULT_RISK_MIN_RISK_REWARD_RATIO,
        ),
    )

    level = os.environ.get("LOG_LEVEL", defaults.DEFAULT_LOG_LEVEL).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigurationError(f"Invalid log level: {level}")
    logging_cfg = LoggingConfig(
        level=cast(Literal["DEBUG", "INFO", "WARNING", "ERROR"], level)
    )

    ratelimit = RateLimitConfig(
        orders=RateLimitBucketConfig(
            capacity=_get_optional_int(
                "RATE_LIMIT_ORDERS_CAPACITY", defaults.DEFAULT_ORDER_RATE_LIMIT_CAPACITY
            ),
            refill_rate_per_sec=_get_optional_float(
                "RATE_LIMIT_ORDERS_REFILL_PER_SEC",
                defaults.DEFAULT_ORDER_RATE_LIMIT_REFILL_PER_SEC,
            ),
        ),
        rest=RateLimitBucketConfig(
            capacity=_get_optional_int(
                "RATE_LIMIT_REST_CAPACITY", defaults.DEFAULT_REST_RATE_LIMIT_CAPACITY
            ),
            refill_rate_per_sec=_get_optional_float(
                "RATE_LIMIT_REST_REFILL_PER_SEC",
                defaults.DEFAULT_REST_RATE_LIMIT_REFILL_PER_SEC,
            ),
        ),
        hist=RateLimitBucketConfig(
            capacity=_get_optional_int(
                "RATE_LIMIT_HIST_CAPACITY", defaults.DEFAULT_HIST_RATE_LIMIT_CAPACITY
            ),
            refill_rate_per_sec=_get_optional_float(
                "RATE_LIMIT_HIST_REFILL_PER_SEC",
                defaults.DEFAULT_HIST_RATE_LIMIT_REFILL_PER_SEC,
            ),
        ),
    )

    quote_stale_threshold_ms = _get_optional_int(
        "QUOTE_STALE_THRESHOLD_MS", defaults.QUOTE_STALE_THRESHOLD_MS
    )

    telegram_token = _optional_env("TELEGRAM__BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
    telegram_chat_raw = _optional_env("TELEGRAM__CHAT_ID", "TELEGRAM_CHAT_ID")
    telegram_chat_id: int | None = None
    if telegram_chat_raw is not None:
        try:
            telegram_chat_id = int(telegram_chat_raw)
        except ValueError as exc:
            raise ConfigurationError(
                "TELEGRAM__CHAT_ID must be a valid integer"
            ) from exc
    else:
        allowed_ids: list[int] = []
        for key in ("TELEGRAM_ALLOWED_ID", "TELEGRAM_ALLOWED_IDS"):
            for raw in get_csv(key):
                if not raw:
                    continue
                try:
                    allowed_ids.append(int(raw))
                except ValueError as exc:
                    raise ConfigurationError(
                        f"Invalid integer in {key}: {raw!r}"
                    ) from exc
        if allowed_ids:
            if len(allowed_ids) > 1:
                raise ConfigurationError(
                    "Only a single TELEGRAM_ALLOWED_ID is supported; received multiple "
                    "values."
                )
            telegram_chat_id = allowed_ids[0]

    if (telegram_token and telegram_chat_id is None) or (
        telegram_chat_id is not None and not telegram_token
    ):
        raise ConfigurationError(
            "Both TELEGRAM__BOT_TOKEN and TELEGRAM__CHAT_ID must be set to enable "
            "Telegram."
        )

    webhook_url_value = _optional_env("TELEGRAM__WEBHOOK_URL", "TELEGRAM_WEBHOOK_URL")
    webhook_url = (
        HTTP_URL_ADAPTER.validate_python(webhook_url_value)
        if webhook_url_value
        else None
    )
    webhook_path = (
        _optional_env("TELEGRAM__WEBHOOK_PATH", "TELEGRAM_WEBHOOK_PATH")
        or defaults.DEFAULT_TELEGRAM_WEBHOOK_PATH
    )
    webhook_secret = _optional_env(
        "TELEGRAM__WEBHOOK_SECRET", "TELEGRAM_WEBHOOK_SECRET"
    )
    webhook_max_failures = _get_optional_int(
        "TELEGRAM__WEBHOOK_MAX_FAILURES",
        defaults.DEFAULT_TELEGRAM_WEBHOOK_MAX_FAILURES,
    )
    enable_polling_default = _get_optional_bool(
        "TELEGRAM_ENABLE_POLLING_FALLBACK",
        False,
    )
    enable_polling_fallback = _get_optional_bool(
        "TELEGRAM__ENABLE_POLLING_FALLBACK",
        enable_polling_default,
    )
    polling_interval_seconds = _get_optional_float(
        "TELEGRAM__POLLING_INTERVAL_SECONDS",
        defaults.DEFAULT_TELEGRAM_POLLING_INTERVAL_SECONDS,
    )
    webhook_listen_host = (
        _optional_env(
            "TELEGRAM__WEBHOOK_HOST",
            "TELEGRAM_WEBHOOK_HOST",
        )
        or defaults.DEFAULT_TELEGRAM_WEBHOOK_LISTEN_HOST
    )
    webhook_listen_port = _get_optional_int(
        "TELEGRAM__WEBHOOK_PORT",
        int(os.environ.get("PORT", defaults.DEFAULT_TELEGRAM_WEBHOOK_LISTEN_PORT)),
    )

    telegram_cfg = TelegramConfig(
        bot_token=telegram_token,
        chat_id=telegram_chat_id,
        webhook_url=webhook_url,
        webhook_path=webhook_path,
        webhook_secret_token=webhook_secret,
        webhook_max_failures=webhook_max_failures,
        enable_polling_fallback=enable_polling_fallback,
        polling_interval_seconds=polling_interval_seconds,
        webhook_listen_host=webhook_listen_host,
        webhook_listen_port=webhook_listen_port,
    )

    return AppConfig(
        broker=broker,
        risk=risk,
        logging=logging_cfg,
        ratelimit=ratelimit,
        quote_stale_threshold_ms=quote_stale_threshold_ms,
        telegram=telegram_cfg,
    )


__all__ = ["load_from_env", "build_broker_config"]
