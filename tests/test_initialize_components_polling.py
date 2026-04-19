from __future__ import annotations

import types
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
from nifty_scalper_bot.core.app import initialize_components
from nifty_scalper_bot.data.symbols import NIFTY_SPOT_CANONICAL_SYMBOL


class DummyBroker:
    def __init__(self, api_key: str, api_secret: str, access_token: str) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.resolver = None

    def attach_resolver(self, resolver) -> None:  # noqa: ANN001
        self.resolver = resolver


class DummyResolver:
    def __init__(self, broker_client) -> None:  # noqa: ANN001
        self.broker_client = broker_client
        self.warmed = False
        self._mapping = {
            NIFTY_SPOT_CANONICAL_SYMBOL: 111,
            "NIFTY": 111,
            "BANKNIFTY": 222,
        }

    def warm(self) -> None:
        self.warmed = True

    def resolve_many(self, symbols):  # noqa: ANN001
        return [self._mapping.get(str(symbol).upper()) for symbol in symbols]

    def resolve(self, symbol: str):  # noqa: ANN001
        return self._mapping.get(str(symbol).upper())


class DummyMarketDataManager:
    def __init__(self, broker_client, ws_manager, *, resolver) -> None:  # noqa: ANN001
        self.broker_client = broker_client
        self.ws_manager = ws_manager
        self.resolver = resolver
        self._ws_connected = ws_manager is not None
        self._handle_tick_calls: list[dict[str, object]] = []
        self._symbol_by_token: dict[int, str] = {}
        self._heartbeat_callbacks: list[callable] = []  # type: ignore[var-annotated]

    def _handle_tick(self, tick):  # noqa: ANN001
        self._handle_tick_calls.append(dict(tick))

    def set_ws_connected(self, connected: bool) -> None:
        self._ws_connected = bool(connected)

    def register_heartbeat_callback(self, callback) -> None:  # noqa: ANN001
        self._heartbeat_callbacks.append(callback)

    def bump_heartbeat(self) -> None:
        return None

    def get_latest_price(self, _symbol: str) -> None:
        return None


class DummyPollingStreamer:
    def __init__(
        self,
        *,
        broker_client,
        on_tick,
        instrument_resolver,
        poll_interval_ms,
        batch_size,
        require_depth,
        warn_on_rate_limit,
    ) -> None:  # noqa: ANN001
        self.broker_client = broker_client
        self.on_tick = on_tick
        self.instrument_resolver = instrument_resolver
        self.poll_interval_ms = poll_interval_ms
        self.batch_size = batch_size
        self.require_depth = require_depth
        self.warn_on_rate_limit = warn_on_rate_limit
        self.started = False
        self.running = False
        self.start_calls = 0
        self.subscriptions: list[list[int]] = []
        self._tokens: set[int] = set()
        self._interval_s = float(poll_interval_ms) / 1000.0
        self._batch_size = batch_size

    def start(self) -> None:
        self.started = True
        self.running = True
        self.start_calls += 1

    def subscribe_tokens(self, tokens):  # noqa: ANN001
        payload = [int(token) for token in tokens]
        self.subscriptions.append(list(payload))
        self._tokens.update(payload)

    def unsubscribe(self, tokens):  # noqa: ANN001
        for token in tokens:
            self._tokens.discard(int(token))

    def stop(self) -> None:
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def tracked_tokens(self) -> list[int]:
        return sorted(self._tokens)


@pytest.fixture(autouse=True)
def _clear_ws_env(monkeypatch):
    monkeypatch.delenv("WEBSOCKET__DISABLED", raising=False)
    monkeypatch.delenv("WEBSOCKET_DISABLED", raising=False)
    monkeypatch.delenv("STREAM__MODE", raising=False)
    monkeypatch.delenv("STREAMING_MODE", raising=False)
    monkeypatch.delenv("POLLING__ENABLED", raising=False)
    monkeypatch.delenv("POLLING__SYMBOLS", raising=False)


def test_initialize_components_uses_polling_by_default(monkeypatch):
    ctx, dummy_notifier = _initialize_polling_context(monkeypatch)

    assert ctx.websocket_manager is None
    assert ctx.websocket_client is None
    assert isinstance(ctx.streamer, DummyPollingStreamer)
    assert ctx.streamer.started is True
    assert ctx.streamer.start_calls == 1
    assert isinstance(ctx.market_data_manager, DummyMarketDataManager)
    assert ctx.market_data_manager.ws_manager is None
    assert ctx.market_data_manager._ws_connected is False  # type: ignore[attr-defined]
    assert ctx.stream_supervisor is not None
    assert ctx.stream_supervisor.tracked_tokens() == [111]
    assert ctx.stream_supervisor.is_running() is True
    assert ctx.stream_supervisor.status_line().startswith("tokens=1")
    assert dummy_notifier.events == []


def _initialize_polling_context(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    monkeypatch.setattr("nifty_scalper_bot.core.app.ZerodhaKiteClient", DummyBroker)
    monkeypatch.setattr("nifty_scalper_bot.core.app.InstrumentResolver", DummyResolver)
    monkeypatch.setattr(
        "nifty_scalper_bot.core.app.MarketDataManager", DummyMarketDataManager
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.core.app.PollingStreamer", DummyPollingStreamer
    )

    dummy_notifier = types.SimpleNamespace(events=[])

    def _notifier_from_settings(_settings):  # noqa: ANN001
        return dummy_notifier

    monkeypatch.setattr(
        "nifty_scalper_bot.core.app.TelegramEnhancedNotifier.from_settings",
        staticmethod(_notifier_from_settings),
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
        shadow=ShadowSettings(drift_threshold_pct=0.0),
        notifications=NotificationSettings(enabled=False),
        session_allow_out_of_hours=False,
    )

    ctx = initialize_components(settings)
    return ctx, dummy_notifier


def test_poll_tick_normalization_backfills_price(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ctx, _dummy_notifier = _initialize_polling_context(monkeypatch)
    mdm: DummyMarketDataManager = ctx.market_data_manager  # type: ignore[assignment]
    mdm._symbol_by_token[256265] = NIFTY_SPOT_CANONICAL_SYMBOL

    caplog.set_level("WARNING")

    ctx.streamer.on_tick({"token": "256265", "last_price": 195.5})  # type: ignore[attr-defined]

    assert mdm._handle_tick_calls, "Expected MarketDataManager to receive tick"
    tick = mdm._handle_tick_calls[-1]
    assert tick["instrument_token"] == 256265
    assert tick["ltp"] == 195.5
    assert tick["symbol"] == NIFTY_SPOT_CANONICAL_SYMBOL
    assert tick["source"] == "polling"
    warnings = [record.message for record in caplog.records]
    assert "poll_tick_missing_price" not in warnings


def test_poll_tick_invalid_token_is_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ctx, _dummy_notifier = _initialize_polling_context(monkeypatch)
    mdm: DummyMarketDataManager = ctx.market_data_manager  # type: ignore[assignment]

    caplog.set_level("WARNING")

    ctx.streamer.on_tick({"token": "abc", "close": 10.0})  # type: ignore[attr-defined]

    assert mdm._handle_tick_calls, "Expected MarketDataManager to receive tick"
    tick = mdm._handle_tick_calls[-1]
    assert "instrument_token" not in tick
    warnings = [record.message for record in caplog.records]
    assert "invalid_token_value" in warnings
