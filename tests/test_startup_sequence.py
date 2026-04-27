from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nifty_scalper_bot.config.base import (
    AppConfig,
    BrokerConfig,
    LoggingConfig,
    RateLimitBucketConfig,
    RateLimitConfig,
    RiskConfig,
)
from nifty_scalper_bot.config.settings import (
    NotificationSettings,
    OrderSettings,
    RiskSettings,
    Settings,
    ShadowSettings,
    StreamerSettings,
)
from nifty_scalper_bot.core.app import BotContext, TradingSessionGuard, startup_sequence
from nifty_scalper_bot.infra.health import HealthState, create_health_app
from nifty_scalper_bot.utils.errors import ConfigurationError
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


class FailingBroker:
    def __init__(self) -> None:
        self.profile_calls = 0

    def get_profile(self) -> dict[str, Any]:
        self.profile_calls += 1
        raise ConfigurationError("Zerodha access denied")

    def load_instruments(self, _exchange: str) -> None:  # pragma: no cover - defensive
        raise AssertionError("load_instruments should not be called when broker fails")

    def get_positions(self) -> list[dict[str, Any]]:
        return []


class DummyStreamer:
    def __init__(self) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    def is_connected(self) -> bool:
        return self.started

    def backlog_size(self) -> int:
        return 0


class DummyMarketDataManager:
    def __init__(self) -> None:
        self.started = False
        self.start_calls = 0
        self.defer_ws_args: list[bool] = []
        self.ws_start_calls = 0

    def start(self, defer_ws: bool = False) -> None:
        self.started = True
        self.start_calls += 1
        self.defer_ws_args.append(bool(defer_ws))

    def start_websocket(self) -> None:
        self.ws_start_calls += 1

    def stop(self) -> None:
        self.started = False

    def get_latest_tick(self, _symbol: str) -> None:  # pragma: no cover - defensive
        return None

    def get_latest_price(self, _symbol: str) -> None:  # pragma: no cover - defensive
        return None

    async def wait_until_ready(self, timeout: float = 30.0) -> None:
        del timeout
        return None


class DummyOrderManager:
    def __init__(self) -> None:
        self.monitoring_started = False

    def start_monitoring(self) -> None:
        self.monitoring_started = True

    def stop_monitoring(self) -> None:
        self.monitoring_started = False

    def cancel_order(self, _order_id: str) -> None:  # pragma: no cover - defensive
        return None


class DummySafeOrderManager:
    def __init__(self) -> None:
        self.settings = type("_SafeSettings", (), {"enable_live": False})()

    def set_live_enabled(self, enabled: bool) -> None:  # pragma: no cover - defensive
        self.settings.enable_live = enabled

    def throttled_count(self) -> int:
        return 0

    def rejection_count(self) -> int:
        return 0

    def stop_monitoring(self) -> None:  # pragma: no cover - defensive
        return None

    def start_monitoring(self) -> None:  # pragma: no cover - defensive
        return None


class DummyRiskManager:
    def is_green(self) -> bool:
        return True

    def force_shadow(self, _flag: bool) -> None:  # pragma: no cover - defensive
        return None

    def record_rejection(self, _reason: str) -> None:  # pragma: no cover - defensive
        return None

    def cooldown_remaining(self) -> float:
        return 0.0


class DummyPositionManager:
    def get_all_positions(self) -> list[Any]:
        return []

    def get_pending_orders(self) -> list[Any]:
        return []

    def save_state(self) -> None:  # pragma: no cover - defensive
        return None


class DummyInstrumentManager:
    def __init__(self) -> None:
        self._kite = None

    def load(self) -> None:
        return None

    def size(self) -> int:
        return 0

    def is_loaded(self) -> bool:
        return False


class DummyStrategyRunner:
    def __init__(self) -> None:
        self.started = False
        self.paused = False
        self.symbols: list[str] = []

    def add_symbol(self, symbol: str) -> None:
        self.symbols.append(symbol)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:  # pragma: no cover - defensive
        self.started = False

    def pause_trading(self) -> None:
        self.paused = True

    def get_status(self) -> dict[str, Any]:
        return {"running": self.started, "active_symbols": list(self.symbols)}


class DummyNotifier:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any] | None]] = []

    async def send_event(
        self, event: str, payload: dict[str, Any] | None = None
    ) -> None:
        self.events.append((event, payload))


@pytest.mark.asyncio
async def test_startup_continues_when_broker_denied() -> None:
    app_config = AppConfig(
        broker=BrokerConfig(
            api_key="demo",
            api_secret="secret",
            access_token="token",
            base_url="https://example.com",
            websocket_url="wss://example.com/ws",
        ),
        risk=RiskConfig(),
        logging=LoggingConfig(level="INFO"),
        ratelimit=RateLimitConfig(
            orders=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
            rest=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
            hist=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
        ),
        quote_stale_threshold_ms=1_000,
    )

    settings = Settings(
        app=app_config,
        enable_live=False,
        orders=OrderSettings(),
        risk=RiskSettings(),
        streamer=StreamerSettings(),
        shadow=ShadowSettings(),
        notifications=NotificationSettings(enabled=False),
        session_allow_out_of_hours=False,
        allow_offmarket_trading=False,
    )

    broker = FailingBroker()
    streamer = DummyStreamer()
    mdm = DummyMarketDataManager()
    order_manager = DummyOrderManager()
    safe_order_manager = DummySafeOrderManager()
    risk_manager = DummyRiskManager()
    strategy_runner = DummyStrategyRunner()
    position_manager = DummyPositionManager()

    rate_limiter = RateLimiter()
    guard = TradingSessionGuard(rate_limiter=rate_limiter, risk_manager=risk_manager)
    guard.mark_session_valid()
    pre_status = guard.evaluate()
    assert pre_status.session_valid

    health_state = HealthState(
        streamer=streamer,
        order_manager=safe_order_manager,
        risk_manager=risk_manager,
        live_enabled=lambda: False,
        session_guard=guard,
    )

    notifier = DummyNotifier()

    ctx = BotContext(
        settings=settings,
        config=app_config,
        rate_limiter=rate_limiter,
        broker_client=broker,
        websocket_client=object(),
        websocket_manager=object(),
        streamer=streamer,
        stream_supervisor=None,
        polling_fallback_streamer=None,
        market_data_manager=mdm,
        indicator_engine=object(),
        position_manager=position_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        safe_order_manager=safe_order_manager,
        strategy_manager=object(),
        strategy_runner=strategy_runner,
        instrument_manager=object(),
        shadow_mode_enabled=True,
        shadow_trader=None,
        out_of_hours_override=False,
        telegram_bot=None,
        telegram_application=None,
        telegram_notifier=notifier,
        health_app=create_health_app(health_state),
        message_bus=None,
        session_guard=guard,
    )

    await startup_sequence(ctx)

    assert not streamer.started
    assert mdm.started
    assert mdm.start_calls == 1
    assert mdm.defer_ws_args == [True]
    assert not strategy_runner.started

    status = guard.last_status()
    assert status is not None

    events = notifier.events
    assert events
    assert events[0][0] == "BOT_STARTED"


class HealthyBroker:
    def __init__(self) -> None:
        self.profile_calls = 0

    def get_profile(self) -> dict[str, Any]:
        self.profile_calls += 1
        return {"user_name": "ok"}

    def load_instruments(self, _exchange: str) -> None:
        return None

    def get_positions(self) -> list[dict[str, Any]]:
        return []


@pytest.mark.asyncio
async def test_startup_does_not_double_start_market_data_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nifty_scalper_bot.core.app._hydrate_positions",
        lambda **_kwargs: [],
    )
    app_config = AppConfig(
        broker=BrokerConfig(
            api_key="demo",
            api_secret="secret",
            access_token="token",
            base_url="https://example.com",
            websocket_url="wss://example.com/ws",
        ),
        risk=RiskConfig(),
        logging=LoggingConfig(level="INFO"),
        ratelimit=RateLimitConfig(
            orders=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
            rest=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
            hist=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
        ),
        quote_stale_threshold_ms=1_000,
    )
    settings = Settings(
        app=app_config,
        enable_live=False,
        orders=OrderSettings(),
        risk=RiskSettings(),
        streamer=StreamerSettings(),
        shadow=ShadowSettings(),
        notifications=NotificationSettings(enabled=False),
        session_allow_out_of_hours=False,
        allow_offmarket_trading=False,
    )
    broker = HealthyBroker()
    mdm = DummyMarketDataManager()
    ctx = BotContext(
        settings=settings,
        config=app_config,
        rate_limiter=RateLimiter(),
        broker_client=broker,
        websocket_client=None,
        websocket_manager=None,
        streamer=DummyStreamer(),
        stream_supervisor=None,
        polling_fallback_streamer=None,
        market_data_manager=mdm,
        indicator_engine=object(),
        position_manager=DummyPositionManager(),
        risk_manager=DummyRiskManager(),
        order_manager=DummyOrderManager(),
        safe_order_manager=DummySafeOrderManager(),
        strategy_manager=object(),
        strategy_runner=DummyStrategyRunner(),
        instrument_manager=DummyInstrumentManager(),
        shadow_mode_enabled=True,
        shadow_trader=None,
        out_of_hours_override=False,
        telegram_bot=None,
        telegram_application=None,
        telegram_notifier=DummyNotifier(),
        health_app=create_health_app(
            HealthState(
                streamer=DummyStreamer(),
                order_manager=DummySafeOrderManager(),
                risk_manager=DummyRiskManager(),
                live_enabled=lambda: False,
            )
        ),
        message_bus=None,
        session_guard=TradingSessionGuard(
            rate_limiter=RateLimiter(),
            risk_manager=DummyRiskManager(),
        ),
        websocket_enabled=False,
    )
    await startup_sequence(ctx)
    assert mdm.start_calls == 1
    assert mdm.defer_ws_args == [True]


def test_settings_execution_mode_and_data_dir_compat() -> None:
    app_config = AppConfig(
        broker=BrokerConfig(
            api_key='demo',
            api_secret='secret',
            access_token='token',
            base_url='https://example.com',
            websocket_url='wss://example.com/ws',
        ),
        risk=RiskConfig(),
        logging=LoggingConfig(level='INFO'),
        ratelimit=RateLimitConfig(
            orders=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
            rest=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
            hist=RateLimitBucketConfig(capacity=10, refill_rate_per_sec=10.0),
        ),
        quote_stale_threshold_ms=1_000,
    )

    settings = Settings(
        app=app_config,
        enable_live=False,
        orders=OrderSettings(),
        risk=RiskSettings(),
        streamer=StreamerSettings(),
        shadow=ShadowSettings(),
        notifications=NotificationSettings(enabled=False),
        session_allow_out_of_hours=False,
        allow_offmarket_trading=False,
    )

    assert settings.execution_mode == 'SHADOW'
    assert settings.data_dir == settings.replay.data_dir

    settings.paper_mode = True
    assert settings.execution_mode == 'PAPER'

    settings.paper_mode = False
    settings.enable_live = True
    assert settings.execution_mode == 'LIVE'


def test_startup_logging_uses_configured_and_effective_mode_fields() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'Startup | configured_mode=%s | effective_mode=%s | live_orders_armed=%s | trading_ready=%s' in source


def test_startup_blocks_live_on_quote_access_denied() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'LIVE_TRADING_BLOCKED reason=broker_quote_access_denied' in source
    assert 'BROKER_QUOTE_CAPABILITY status=unavailable reason=%s' in source


def test_startup_emits_market_session_and_data_warmup_logs() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'MARKET_SESSION_STATE state=%s' in source
    assert 'DATA_WARMUP reason=%s' in source


def test_resolve_quote_capability_combines_mdm_and_broker() -> None:
    from nifty_scalper_bot.core.app import _resolve_quote_capability

    class _MDMOk:
        def quote_api_status_snapshot(self) -> dict[str, Any]:
            return {"available": True, "error": None}

    class _BrokerForbidden:
        def quote_api_status_snapshot(self) -> dict[str, Any]:
            return {"available": False, "error": "access_denied"}

    class _Ctx:
        market_data_manager = _MDMOk()
        broker_client = _BrokerForbidden()

    snap = _resolve_quote_capability(_Ctx())
    assert snap["available"] is False
    assert snap["error"] == "access_denied"


def test_resolve_quote_capability_unwraps_robust_provider() -> None:
    from nifty_scalper_bot.core.app import _resolve_quote_capability

    class _Inner:
        def quote_api_status_snapshot(self) -> dict[str, Any]:
            return {"available": False, "error": "access_denied"}

    class _RobustProvider:
        _broker = _Inner()

    class _MDMOk:
        def quote_api_status_snapshot(self) -> dict[str, Any]:
            return {"available": True, "error": None}

    class _Ctx:
        market_data_manager = _MDMOk()
        broker_client = _RobustProvider()

    snap = _resolve_quote_capability(_Ctx())
    assert snap["available"] is False
    assert snap["error"] == "access_denied"


def test_resolve_quote_capability_margin_success_does_not_override_quote_failure() -> None:
    from nifty_scalper_bot.core.app import _resolve_quote_capability

    class _BrokerWithDeniedQuote:
        def quote_api_status_snapshot(self) -> dict[str, Any]:
            return {"available": False, "error": "access_denied"}

        def get_margins(self, *_args, **_kwargs) -> dict[str, Any]:
            return {"equity": {"available": {"cash": 1000.0}}}

    class _Ctx:
        market_data_manager = None
        broker_client = _BrokerWithDeniedQuote()

    snap = _resolve_quote_capability(_Ctx())
    assert snap["available"] is False
    assert snap["error"] == "access_denied"
