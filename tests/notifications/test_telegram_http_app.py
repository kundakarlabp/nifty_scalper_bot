from __future__ import annotations

import importlib
from collections.abc import Iterable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot.config import settings as settings_module


def _ensure_routes(app: FastAPI) -> Iterable[str]:
    return [route.path for route in app.router.routes]


def test_http_app_registers_webhook(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_API_KEY", "demo")
    monkeypatch.setenv("BROKER_API_SECRET", "demo-secret")
    monkeypatch.setenv("BROKER_ACCESS_TOKEN", "demo-token")
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_PUBLIC_BASE_URL", "https://example.com")

    settings_module.get_settings.cache_clear()

    core_app = importlib.import_module("nifty_scalper_bot.core.app")
    importlib.reload(core_app)

    events: list[tuple[str, dict[str, object] | None]] = []

    async def fake_send_event(
        self, event: str, payload: dict[str, object] | None = None
    ) -> None:
        events.append((event, payload))

    monkeypatch.setattr(
        core_app.TelegramEnhancedNotifier,
        "send_event",
        fake_send_event,
        raising=False,
    )

    captured: dict[str, object] = {}

    async def fake_set_webhook(
        self,
        url: str,
        drop_pending_updates: bool = True,
        allowed_updates: list[str] | None = None,
    ) -> bool:
        captured["url"] = url
        captured["drop_pending"] = drop_pending_updates
        captured["allowed_updates"] = allowed_updates
        return True

    monkeypatch.setattr("telegram.Bot.set_webhook", fake_set_webhook, raising=False)

    async def fail_activate(
        self, reason: str, *, interval: float | None = None
    ) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError(
            "fallback polling should not start on successful registration"
        )

    monkeypatch.setattr(
        core_app.TelegramWebhookController,
        "activate_polling_fallback",
        fail_activate,
    )

    app = core_app.get_http_app()

    assert "/telegram/webhook" in _ensure_routes(app)

    notifier = core_app.get_telegram_notifier()
    assert notifier is not None

    with TestClient(app):
        pass

    assert captured["url"] == "https://example.com/telegram/webhook"
    assert captured["allowed_updates"] is None
    assert captured["drop_pending"] is False
    assert events == []

    settings_module.get_settings.cache_clear()
    core_app._HTTP_APP = None
    core_app._HTTP_CONTROLLER = None
    core_app._HTTP_NOTIFIER = None


def test_http_app_skips_webhook_controller_when_webhook_missing(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_API_KEY", "demo")
    monkeypatch.setenv("BROKER_API_SECRET", "demo-secret")
    monkeypatch.setenv("BROKER_ACCESS_TOKEN", "demo-token")
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_ALLOW_POLL_FALLBACK", "true")
    monkeypatch.delenv("TELEGRAM_PUBLIC_BASE_URL", raising=False)

    settings_module.get_settings.cache_clear()

    core_app = importlib.import_module("nifty_scalper_bot.core.app")
    importlib.reload(core_app)

    async def fail_activate(
        self, reason: str, *, interval: float | None = None
    ) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError(f"polling fallback should not start (reason={reason})")

    def fail_register(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("webhook registration should not be attempted")

    monkeypatch.setattr(core_app, "register_webhook", fail_register)
    monkeypatch.setattr(
        core_app.TelegramWebhookController,
        "activate_polling_fallback",
        fail_activate,
    )

    app = core_app.get_http_app()

    assert "/telegram/webhook" not in _ensure_routes(app)
    assert core_app._HTTP_CONTROLLER is None

    with TestClient(app):
        pass

    settings_module.get_settings.cache_clear()
    core_app._HTTP_APP = None
    core_app._HTTP_CONTROLLER = None
    core_app._HTTP_NOTIFIER = None


def test_http_app_skips_webhook_controller_when_webhook_disabled(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_API_KEY", "demo")
    monkeypatch.setenv("BROKER_API_SECRET", "demo-secret")
    monkeypatch.setenv("BROKER_ACCESS_TOKEN", "demo-token")
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_ENABLE_POLLING_FALLBACK", "true")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_ENABLED", "false")
    monkeypatch.delenv("TELEGRAM_PUBLIC_BASE_URL", raising=False)

    settings_module.get_settings.cache_clear()

    core_app = importlib.import_module("nifty_scalper_bot.core.app")
    importlib.reload(core_app)

    async def fail_activate(
        self, reason: str, *, interval: float | None = None
    ) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError(f"polling fallback should not start (reason={reason})")

    def fail_register(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("webhook registration should be skipped when disabled")

    monkeypatch.setattr(
        core_app.TelegramWebhookController,
        "activate_polling_fallback",
        fail_activate,
    )
    monkeypatch.setattr(core_app, "register_webhook", fail_register)

    app = core_app.get_http_app()

    assert "/telegram/webhook" not in _ensure_routes(app)
    assert core_app._HTTP_CONTROLLER is None

    with TestClient(app):
        pass

    settings_module.get_settings.cache_clear()
    core_app._HTTP_APP = None
    core_app._HTTP_CONTROLLER = None
    core_app._HTTP_NOTIFIER = None
