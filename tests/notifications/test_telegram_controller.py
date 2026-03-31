"""Unit tests for Telegram webhook helpers."""

from __future__ import annotations

import asyncio
import json
import time
import types
from datetime import datetime, timezone
from typing import Any, Sequence

import pytest
from fastapi.testclient import TestClient
from telegram.constants import ParseMode

from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.core.unified_manager import UMPlan
from nifty_scalper_bot.execution.margin_engine import MarginDecision
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.notifications import telegram_controller as telegram_module
from nifty_scalper_bot.notifications.telegram_controller import (
    TelegramBot,
    TelegramDeps,
    TelegramRuntimeMetrics,
    TelegramWebhookServer,
    _mask_token,
)


class _FakeBot:
    """Minimal stub implementing the Telegram webhook bot surface."""

    def __init__(self) -> None:
        self._metrics = TelegramRuntimeMetrics()
        self.processed: list[dict[str, Any]] = []
        self.last_failure: dict[str, Any] | None = None
        self._polling = False

    async def process_webhook_payload(self, payload: dict[str, Any]) -> int | None:
        self.processed.append(payload)
        self._metrics.webhook_updates += 1
        return payload.get("chat_id")

    def record_webhook_success(self) -> None:  # pragma: no cover - noop stub
        pass

    def record_webhook_failure(self, *, reason: str, status_code: int) -> None:
        self._metrics.webhook_failures += 1
        self.last_failure = {"reason": reason, "status_code": status_code}

    def record_webhook_secret_failure(self) -> None:
        self._metrics.webhook_secret_failures += 1

    @property
    def metrics(self) -> TelegramRuntimeMetrics:
        return self._metrics

    @property
    def is_polling_active(self) -> bool:
        return self._polling


def test_mask_token_redacts_middle() -> None:
    masked = _mask_token("1234567890ABCDEF")
    assert masked.startswith("1234")
    assert masked.endswith("CDEF")
    assert "5" not in masked


@pytest.mark.asyncio
async def test_reply_plain_mode_renders_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM__PLAIN_TEXT", "true")
    deps = TelegramDeps(token="x", chat_id=1, app_version="test")
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    stub = _StubMessenger()
    monkeypatch.setattr(
        bot,
        "_ensure_messenger",
        lambda ctx, current_chat, bot_override=None: stub,
    )
    ctx = _DummyContext()
    html = "<b>Title</b><br><br><pre>code\nblock</pre><br>&lt;done&gt;"

    await bot._reply(chat, ctx, html, parse_mode=ParseMode.HTML)

    assert stub.calls
    call = stub.calls[-1]
    assert call["parse_mode"] is None
    assert call["text"] == "Title\n\ncode\nblock\n<done>"


@pytest.mark.asyncio
async def test_guard_admin_rejects_unlisted_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEGRAM__ADMIN_CHAT_ID", raising=False)
    deps = TelegramDeps(token="x", chat_id=1, app_version="test")
    bot = TelegramBot(deps)
    chat = _DummyChat(2)
    update = _DummyUpdate(chat)
    stub = _StubMessenger()
    monkeypatch.setattr(
        bot,
        "_ensure_messenger",
        lambda ctx, current_chat, bot_override=None: stub,
    )
    allowed = await bot._guard_admin(update)
    assert not allowed
    assert stub.calls
    assert "elevated" in stub.calls[-1]["text"].lower()


@pytest.mark.asyncio
async def test_guard_admin_allows_allowlisted_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM__ADMIN_CHAT_ID", "2")
    deps = TelegramDeps(token="x", chat_id=1, app_version="test")
    bot = TelegramBot(deps)
    chat = _DummyChat(2)
    update = _DummyUpdate(chat)
    assert await bot._guard_admin(update)


def test_webhook_secret_guard_rejects_invalid_token() -> None:
    bot = _FakeBot()
    server = TelegramWebhookServer(
        bot=bot,
        path="/hook",
        secret_token="expected",
        host="127.0.0.1",
        port=0,
    )
    client = TestClient(server.app)

    response = client.post(
        "/hook",
        json={"update_id": 1},
        headers={"X-Telegram-Bot-Api-Secret-Token": "invalid"},
    )

    assert response.status_code == 401
    assert bot.metrics.webhook_secret_failures == 1
    assert bot.last_failure is None


def test_webhook_processes_valid_payload() -> None:
    bot = _FakeBot()
    server = TelegramWebhookServer(
        bot=bot,
        path="/hook",
        secret_token=None,
        host="127.0.0.1",
        port=0,
    )
    client = TestClient(server.app)

    response = client.post("/hook", json={"update_id": 42, "chat_id": 7})

    assert response.status_code == 200
    assert bot.metrics.webhook_updates == 1
    assert bot.last_failure is None
    assert bot.processed[-1]["update_id"] == 42


def test_webhook_invalid_json_records_failure() -> None:
    bot = _FakeBot()
    server = TelegramWebhookServer(
        bot=bot,
        path="/hook",
        secret_token=None,
        host="127.0.0.1",
        port=0,
    )
    client = TestClient(server.app)

    response = client.post(
        "/hook",
        data="not-json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert bot.last_failure == {"reason": "invalid_json", "status_code": 400}


class _DummyMessage:
    def __init__(self, chat: "_DummyChat", text: str) -> None:
        self.chat = chat
        self.text = text
        self.edits: list[str] = []

    async def edit_text(self, text: str) -> "_DummyMessage":
        self.edits.append(text)
        self.chat.sent.append(text)
        self.text = text
        return self


class _DummyChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id
        self.sent: list[str] = []
        self.messages: list[_DummyMessage] = []

    async def send_message(
        self, text: str, parse_mode: str | None = None, **_: Any
    ) -> _DummyMessage:
        message = _DummyMessage(self, text)
        self.sent.append(text)
        self.messages.append(message)
        return message


class _DummyContext:
    def __init__(self, args: list[str] | None = None) -> None:
        self.args = args or []


class _DummyUpdate:
    def __init__(self, chat: _DummyChat) -> None:
        self.effective_chat = chat


class _StubMessenger:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def send_text(
        self,
        chat_id: int,
        text: str,
        *,
        html_ready: bool = False,
        parse_mode=None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "chat_id": chat_id,
                "text": text,
                "html_ready": html_ready,
                "parse_mode": parse_mode,
                "kwargs": dict(kwargs),
            }
        )
        return types.SimpleNamespace(text=text)


@pytest.mark.asyncio
async def test_alert_aggregator_batches_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = TelegramDeps(token="x", chat_id=99, app_version="test")
    bot = TelegramBot(deps)
    stub = _StubMessenger()
    bot._app = types.SimpleNamespace(bot=object())
    monkeypatch.setattr(
        bot,
        "_ensure_messenger",
        lambda ctx, current_chat, bot_override=None: stub,
    )
    bot._aggregation_interval = 0.2
    bot._alert_queue = asyncio.Queue()
    aggregator_task = asyncio.create_task(bot._alert_aggregator())
    try:
        await bot._alert_queue.put(("log:error", "First failure", "warning", False))
        await asyncio.sleep(0.05)
        await bot._alert_queue.put(("log:error", "Second failure", "warning", False))
        await asyncio.sleep(0.35)
    finally:
        aggregator_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await aggregator_task

    assert stub.calls
    payload = stub.calls[-1]
    assert payload["chat_id"] == deps.chat_id
    text = payload["text"]
    assert text.startswith("⚠️ Aggregated alerts")
    assert "Breakdown:" in text
    assert "log" in text.lower()
    assert not bot._aggregated_alerts


class _DummyRunner:
    def __init__(self) -> None:
        self.resume_calls = 0
        self.pause_calls = 0
        self.paused = False

    def resume_trading(self) -> None:
        self.resume_calls += 1
        self.paused = False

    def pause_trading(self) -> None:
        self.pause_calls += 1
        self.paused = True

    def get_status(self) -> dict[str, Any]:
        return {"active_symbols": ["NIFTY"], "trading_paused": self.paused}


class _GuardStatus:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _DummyGuard:
    def __init__(
        self,
        evaluate_data: dict[str, Any] | None = None,
        *,
        last_status_data: dict[str, Any] | None = None,
    ) -> None:
        self._evaluate_data = evaluate_data or {}
        self._last_status_data = last_status_data or self._evaluate_data

    def evaluate(self) -> _GuardStatus:
        return _GuardStatus(self._evaluate_data)

    def last_status(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(**self._last_status_data)


class _DummyWS:
    def __init__(self) -> None:
        self.refresh_called = 0
        self.force_called = 0
        self._last_error = types.SimpleNamespace(code=1006, reason="handshake_timeout")

    def refresh_session(self) -> None:
        self.refresh_called += 1

    def force_reconnect(self) -> None:
        self.force_called += 1

    def last_error(self) -> Any:
        return self._last_error


class _DummyPollingStreamer:
    def __init__(self) -> None:
        self.tokens: set[int] = set()

    def subscribe_tokens(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            self.tokens.add(int(token))

    def unsubscribe(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            self.tokens.discard(int(token))

    def tracked_tokens(self) -> list[int]:
        return sorted(self.tokens)


class _DummySupervisor:
    def __init__(
        self,
        *,
        running: bool = True,
        interval: float = 0.7,
        batch: int = 200,
        mapping: dict[str, int] | None = None,
    ) -> None:
        self.tokens: set[int] = set()
        self.running = running
        self.interval = interval
        self.batch = batch
        self.mapping = {"NIFTY": 123, "BANKNIFTY": 456}
        if mapping:
            self.mapping.update(
                {key.upper(): int(value) for key, value in mapping.items()}
            )
        self.ensure_started_calls = 0

    def subscribe_tokens(self, tokens: Sequence[int]) -> int:
        for token in tokens:
            self.tokens.add(int(token))
        return len(self.tokens)

    def unsubscribe_tokens(self, tokens: Sequence[int]) -> int:
        for token in tokens:
            self.tokens.discard(int(token))
        return len(self.tokens)

    def is_running(self) -> bool:
        return self.running

    def resolve_symbols(
        self, symbols: Sequence[str]
    ) -> tuple[list[int], list[str], dict[str, int]]:
        resolved: list[int] = []
        unresolved: list[str] = []
        mapping: dict[str, int] = {}
        for symbol in symbols:
            key = symbol.upper()
            token = self.mapping.get(key)
            if token is None:
                unresolved.append(key)
            else:
                resolved.append(int(token))
                mapping[key] = int(token)
        return resolved, unresolved, mapping

    def token_count(self) -> int:
        return len(self.tokens)

    def tracked_tokens(self) -> list[int]:
        return sorted(self.tokens)

    def ensure_started(self) -> None:
        self.ensure_started_calls += 1

    def status_line(self) -> str:
        heart = "💓 running" if self.running else "⛔ stopped"
        return f"tokens={len(self.tokens)} @{self.interval:.2f}s {heart}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "status": self.status_line(),
            "tokens": len(self.tokens),
            "interval_s": self.interval,
            "batch_size": self.batch,
        }


class _DummyMDM:
    def stats(self) -> dict[str, Any]:
        return {"ws_connected": False, "symbols": 0}


@pytest.mark.asyncio
async def test_cmd_start_resumes_runner() -> None:
    runner = _DummyRunner()
    guard = _DummyGuard(
        {
            "session_valid": True,
            "rate_limits_ok": True,
            "market_open": True,
            "risk_green": True,
            "reasons": [],
            "override_out_of_hours": False,
        }
    )
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        strategy_runner=runner,
        session_guard=guard,
        get_shadow_mode=lambda: True,
        set_shadow_mode=lambda _: True,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    await bot.cmd_start(update, ctx)

    assert runner.resume_calls == 1
    message = chat.sent[-1]
    assert "trading resumed" in message
    assert "mode=shadow" in message
    assert "symbols=NIFTY" in message


@pytest.mark.asyncio
async def test_cmd_pause_and_resume_toggle_trading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = TelegramDeps(token="x", chat_id=1, app_version="test")
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = _DummyUpdate(chat)
    stub = _StubMessenger()
    monkeypatch.setattr(
        bot,
        "_ensure_messenger",
        lambda ctx, current_chat, bot_override=None: stub,
    )

    await bot.cmd_pause(update, _DummyContext())
    assert stub.calls
    assert stub.calls[-1]["text"].startswith("⏸️")
    assert not trading_switch().can_trade()

    stub.calls.clear()
    await bot.cmd_resume(update, _DummyContext())
    assert stub.calls
    assert stub.calls[-1]["text"].startswith("▶️")
    assert trading_switch().can_trade()


@pytest.mark.asyncio
async def test_cmd_cooldown_sets_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = TelegramDeps(token="x", chat_id=1, app_version="test")
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = _DummyUpdate(chat)
    stub = _StubMessenger()
    monkeypatch.setattr(
        bot,
        "_ensure_messenger",
        lambda ctx, current_chat, bot_override=None: stub,
    )

    await bot.cmd_cooldown(update, _DummyContext(args=["5"]))
    assert stub.calls
    assert stub.calls[-1]["text"].startswith("⏲️")
    snapshot = trading_switch().snapshot()
    assert not snapshot.can_trade
    assert snapshot.remaining_seconds > 0


@pytest.mark.asyncio
async def test_cmd_stop_pauses_runner() -> None:
    runner = _DummyRunner()
    guard = _DummyGuard(
        {
            "session_valid": True,
            "rate_limits_ok": True,
            "market_open": False,
            "risk_green": True,
            "reasons": ["Outside trading window"],
            "override_out_of_hours": True,
        }
    )
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        strategy_runner=runner,
        session_guard=guard,
        get_shadow_mode=lambda: False,
        set_shadow_mode=lambda _: True,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    await bot.cmd_stop(update, ctx)

    assert runner.pause_calls == 1
    message = chat.sent[-1]
    assert "trading paused" in message
    assert "override=true" in message
    assert "paused=True" in message


@pytest.mark.asyncio
async def test_cmd_ws_reconnect_uses_force_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _DummyWS()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        websocket_manager=ws,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    async def _to_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread)

    await bot.cmd_ws_reconnect(update, ctx)

    assert ws.refresh_called == 1
    assert ws.force_called == 1
    assert any("last_error=1006:handshake_timeout" in msg for msg in chat.sent)


@pytest.mark.asyncio
async def test_cmd_mdm_reports_ws_flag() -> None:
    mdm = _DummyMDM()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        market_data_manager=mdm,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    await bot.cmd_mdm(update, ctx)

    message = chat.sent[-1]
    assert '"ws_connected": false' in message


@pytest.mark.asyncio
async def test_cmd_diag_price_renders_snapshot() -> None:
    class _StubHub:
        def normalize(self, symbol: str) -> str:
            return symbol.upper()

        def get_quote(self, symbol: str, allow_pull: bool = False) -> dict[str, Any]:
            return {"bid": 100.5, "ask": 101.5, "ltp": 101.0}

        def is_fresh(self, symbol: str) -> tuple[bool, dict[str, Any]]:
            return True, {"effective_ms": 120.0, "threshold_ms": 500.0}

    class _StubMDM:
        def mdm_status(self) -> dict[str, Any]:
            return {
                "last_tick_source": {"NIFTY": "feed.primary"},
                "last_tick_age": {"NIFTY": 0.12},
                "fallback_enabled": False,
                "heartbeat_age": 0.5,
            }

        def cached_mid(self, symbol: str) -> tuple[float, float]:
            return 101.0, 85.0

    class _StubRisk:
        def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
            return True, ("ok",)

    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        data_hub=_StubHub(),
        market_data_manager=_StubMDM(),
        risk_manager=_StubRisk(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["nifty"])

    await bot.cmd_diag_price(update, ctx)

    message = chat.sent[-1]
    assert message.startswith("diag_price")
    start = message.index("<pre>") + 5
    end = message.index("</pre>")
    payload = json.loads(message[start:end])
    assert payload["symbol"] == "NIFTY"
    assert payload["fresh_ok"] is True
    assert payload["gate"]["allowed"] is True


@pytest.mark.asyncio
async def test_cmd_diag_risk_renders_snapshot() -> None:
    class _StubSwitches:
        def snapshot(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                day_loss=10.0,
                max_day_loss=50.0,
                consecutive_losses=1,
                max_consecutive_losses=5,
                cooldown_remaining=15.0,
            )

    class _StubRiskManager:
        def __init__(self) -> None:
            self.risk_state = types.SimpleNamespace(
                quote_stale_ms_threshold=2500,
                spread_widen_mult=1.5,
                max_intraday_dd=1500.0,
                max_consecutive_losses=3,
                _median_spread=0.35,
                _consecutive_losses=1,
                _reasons={"cooldown"},
                _data_symbol="NIFTY",
                _last_tick_ns=time.time_ns() - 5_000_000,
            )
            self._switches = _StubSwitches()

        def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
            return False, ("cooldown",)

        def snapshot(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                daily_realized=5.0,
                daily_loss_limit=50.0,
                day_loss=3.5,
                max_day_loss=15.0,
                losses_in_row=1,
                cooldown_remaining=0.0,
                breaker_tripped=False,
                breaker_reason=None,
                shadow_forced=False,
                per_trade_risk_pct=1.2,
                last_rejection="none",
                timestamp=datetime.now(timezone.utc),
            )

    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        risk_manager=_StubRiskManager(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext([])

    await bot.cmd_diag_risk(update, ctx)

    message = chat.sent[-1]
    assert message.startswith("diag_risk")
    start = message.index("<pre>") + 5
    end = message.index("</pre>")
    payload = json.loads(message[start:end])
    assert payload["gate"]["allowed"] is False
    assert payload["risk_state"]["reasons"] == ["cooldown"]
    assert payload["switches"]["cooldown_remaining"] == 15.0


@pytest.mark.asyncio
async def test_cmd_go_flat_cancels_and_flattens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubOrderManager:
        def __init__(self) -> None:
            self.cancel_calls = 0
            self.place_calls: list[tuple[str, str, int]] = []

        def cancel_pending_orders(self) -> list[str]:
            self.cancel_calls += 1
            return ["A1", "B2"]

        def place_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: int,
            price: float | None = None,
        ) -> None:
            self.place_calls.append((symbol, side, quantity))

    class _StubPositionManager:
        def get_open_positions(self) -> list[types.SimpleNamespace]:
            return [
                types.SimpleNamespace(symbol="NIFTY", quantity=2, side="LONG"),
                types.SimpleNamespace(symbol="BANKNIFTY", quantity=-1, side="SHORT"),
            ]

    class _StubMDM:
        def get_latest_price(self, symbol: str) -> float:
            return 100.0 if symbol == "NIFTY" else 200.0

    order_manager = _StubOrderManager()
    position_manager = _StubPositionManager()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        order_manager=order_manager,
        position_manager=position_manager,
        market_data_manager=_StubMDM(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext([])

    stub_messenger = _StubMessenger()
    monkeypatch.setattr(
        bot,
        "_ensure_messenger",
        lambda ctx_obj, current_chat, bot_override=None: stub_messenger,
    )

    await bot.cmd_go_flat(update, ctx)

    assert order_manager.cancel_calls == 0
    assert not order_manager.place_calls
    assert stub_messenger.calls
    first_message = stub_messenger.calls[-1]["text"]
    assert "Confirm emergency flatten" in first_message

    await bot.cmd_confirm(update, _DummyContext(["emergency_stop"]))

    assert order_manager.cancel_calls == 1
    assert len(order_manager.place_calls) == 2
    assert any(
        call[0] == "NIFTY" and call[1] == "SELL" for call in order_manager.place_calls
    )
    assert any(
        call[0] == "BANKNIFTY" and call[1] == "BUY"
        for call in order_manager.place_calls
    )
    assert stub_messenger.calls
    message = stub_messenger.calls[-1]["text"]
    assert "Emergency flatten executed" in message
    assert "Cancelled orders: 2" in message
    assert "Closed legs" in message


@pytest.mark.asyncio
async def test_cmd_plan_reports_margin_block() -> None:
    class _StubUnifiedManager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []

        def plan(self, symbol: str, side: str, qty: int) -> UMPlan:
            self.calls.append((symbol, side, qty))
            return UMPlan(
                ok=False,
                qty=0,
                est_required=150.0,
                reason="MARGIN",
                meta={
                    "price_source": "mdm",
                    "ltp": 150.0,
                    "lot_size": 25,
                    "exchange": "NFO",
                    "tradingsymbol": "NIFTY25OCT25900CE",
                },
            )

    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        unified_manager=_StubUnifiedManager(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["NIFTY", "BUY", "2"])

    await bot.cmd_plan(update, ctx)

    message = chat.sent[-1]
    assert message.startswith("/plan NIFTY BUY 2")
    assert "status=blocked" in message
    assert "est=₹150.00" in message
    assert "source=mdm" in message
    assert "reason=MARGIN" in message


@pytest.mark.asyncio
async def test_cmd_margin_reports_snapshot() -> None:
    class _StubOrderManager:
        def _pre_trade_decision(
            self,
            *,
            symbol: str,
            side: str,
            quantity: int,
            product: str | None,
            price: float | None = None,
            stop_loss: float | None = None,
            atr: float | None = None,
        ) -> tuple[MarginDecision, dict[str, object]]:
            decision = MarginDecision(
                ok=True,
                reason=None,
                order_type="NRML",
                quantity=quantity,
                est_required=75.0,
                available=200.0,
            )
            return decision, {"available_source": "broker"}

    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        order_manager=_StubOrderManager(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["BANKNIFTY", "5"])

    await bot.cmd_margin(update, ctx)

    message = chat.sent[-1]
    assert "💰 Margin for BANKNIFTY BUY x5" in message
    assert "Required: ₹75.00" in message
    assert "Available: ₹200.00 (broker)" in message
    assert "Status: ✅ SUFFICIENT" in message


@pytest.mark.asyncio
async def test_cmd_margin_uses_unified_plan_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubOrderManager:
        def _pre_trade_decision(
            self,
            *,
            symbol: str,
            side: str,
            quantity: int,
            product: str | None,
            price: float | None = None,
            stop_loss: float | None = None,
            atr: float | None = None,
        ) -> tuple[MarginDecision, dict[str, object]]:
            decision = MarginDecision(
                ok=True,
                reason=None,
                order_type="NRML",
                quantity=0,
                est_required=0.0,
                available=50.0,
            )
            return decision, {"available_source": "cache"}

    class _StubUnifiedManager:
        def plan(self, symbol: str, side: str, qty: int) -> UMPlan:
            return UMPlan(
                ok=True,
                qty=3,
                est_required=123.45,
                reason=None,
                meta={"price_source": "est", "ltp": 5.0},
            )

    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        order_manager=_StubOrderManager(),
        unified_manager=_StubUnifiedManager(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["FINOPT", "2"])

    calls: list[str] = []

    def _record_fallback(*, source: str) -> None:
        calls.append(source)

    monkeypatch.setattr(
        METRICS,
        "record_telegram_margin_um_fallback",
        _record_fallback,
        raising=True,
    )

    await bot.cmd_margin(update, ctx)

    message = chat.sent[-1]
    assert "Required: ₹123.45" in message
    assert "Price source: est" in message
    assert "Order type: NRML x3" in message
    assert "ℹ️ Using unified manager estimate; broker margin snapshot stale." in message
    assert calls == ["est"]


@pytest.mark.asyncio
async def test_cmd_subscribe_polling_streamer() -> None:
    streamer = _DummyPollingStreamer()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        streamer=streamer,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["123", "oops", "456"])

    await bot.cmd_subscribe(update, ctx)

    assert streamer.tracked_tokens() == [123, 456]
    message = chat.sent[-1]
    assert "Subscribed 2 token(s)" in message
    assert "Now tracking 2" in message
    assert "symbols ignored: OOPS" in message


@pytest.mark.asyncio
async def test_cmd_subscribe_stream_supervisor() -> None:
    supervisor = _DummySupervisor()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        stream_supervisor=supervisor,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["123", "456"])

    await bot.cmd_subscribe(update, ctx)

    assert supervisor.tokens == {123, 456}
    message = chat.sent[-1]
    assert "Added 2 token(s)." in message
    assert "Poll: tokens=2" in message
    assert supervisor.ensure_started_calls == 1


@pytest.mark.asyncio
async def test_cmd_subscribe_stream_supervisor_resolves_symbols() -> None:
    supervisor = _DummySupervisor(mapping={"NIFTY": 789})
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        stream_supervisor=supervisor,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["NIFTY", "654"])

    await bot.cmd_subscribe(update, ctx)

    assert supervisor.tokens == {789, 654}
    message = chat.sent[-1]
    assert "Resolved: NIFTY->789" in message
    assert "Poll: tokens=2" in message
    assert supervisor.ensure_started_calls == 1


@pytest.mark.asyncio
async def test_cmd_unsubscribe_polling_streamer() -> None:
    streamer = _DummyPollingStreamer()
    streamer.subscribe_tokens([111, 222, 333])
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        streamer=streamer,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["222", "bad"])

    await bot.cmd_unsubscribe(update, ctx)

    assert streamer.tracked_tokens() == [111, 333]
    message = chat.sent[-1]
    assert "Unsubscribed 1 token(s)" in message
    assert "Now tracking 2" in message
    assert "ignored: bad" in message


@pytest.mark.asyncio
async def test_cmd_unsubscribe_stream_supervisor() -> None:
    supervisor = _DummySupervisor()
    supervisor.subscribe_tokens([111, 222, 333])
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        stream_supervisor=supervisor,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["222"])

    await bot.cmd_unsubscribe(update, ctx)

    assert supervisor.tokens == {111, 333}
    message = chat.sent[-1]
    assert "Unsubscribed 1 token(s)" in message
    assert "Now tracking 2" in message
    assert "Poll: tokens=2" in message


@pytest.mark.asyncio
async def test_cmd_poll_status_uses_stream_supervisor() -> None:
    supervisor = _DummySupervisor()
    supervisor.subscribe_tokens([101, 202])
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        stream_supervisor=supervisor,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    await bot.cmd_poll_status(update, ctx)

    message = chat.sent[-1]
    assert "Poll: tokens=2" in message
    assert "💓 running" in message


@pytest.mark.asyncio
async def test_cmd_shadow_off_blocked() -> None:
    guard = _DummyGuard(
        {
            "session_valid": False,
            "rate_limits_ok": True,
            "market_open": False,
            "risk_green": True,
            "reasons": ["Outside trading window"],
            "override_out_of_hours": False,
        }
    )
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        set_shadow_mode=lambda value: value,
        session_guard=guard,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    await bot.cmd_shadow_off(update, ctx)

    message = chat.sent[-1]
    assert message.startswith("shadow=false blocked")
    assert "Outside trading window" in message


@pytest.mark.asyncio
async def test_cmd_shadow_off_override_message() -> None:
    guard = _DummyGuard(
        {
            "session_valid": True,
            "rate_limits_ok": True,
            "market_open": False,
            "risk_green": True,
            "reasons": ["Outside trading window"],
            "override_out_of_hours": True,
        },
        last_status_data={"override_out_of_hours": True},
    )
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        set_shadow_mode=lambda value: not value,
        session_guard=guard,
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    await bot.cmd_shadow_off(update, ctx)

    message = chat.sent[-1]
    assert "override active" in message
    assert "reasons:" in message


@pytest.mark.asyncio
async def test_cmd_paper_lists_sections() -> None:
    state = {"orders": True, "strategy": False}
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        paper_mode_getters={
            "orders": lambda state=state: state["orders"],
            "strategy": lambda state=state: state["strategy"],
        },
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext()

    await bot.cmd_paper(update, ctx)

    message = chat.sent[-1]
    assert "ORDERS" in message
    assert "STRATEGY" in message
    assert "shadow" in message
    assert "live" in message


@pytest.mark.asyncio
async def test_cmd_paper_toggle_updates_state() -> None:
    state = {"orders": True}

    def _setter(value: bool, *, state: dict[str, bool] = state) -> bool:
        state["orders"] = value
        return True

    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        paper_mode_getters={"orders": lambda state=state: state["orders"]},
        paper_mode_setters={"orders": _setter},
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["orders", "live"])

    await bot.cmd_paper(update, ctx)

    assert state["orders"] is False
    message = chat.sent[-1]
    assert message == "orders=live"


@pytest.mark.asyncio
async def test_cmd_fresh_reports_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FreshHub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def is_fresh(self, symbol: str) -> tuple[bool, dict[str, object]]:
            self.calls.append(symbol)
            return True, {
                "effective_ms": 750.0,
                "mono_age_ms": 700.0,
                "server_age_ms": 680.0,
                "threshold_ms": 2000.0,
                "reason": None,
            }

    hub = _FreshHub()
    deps = TelegramDeps(token="x", chat_id=1, app_version="test", data_hub=hub)
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    ctx = _DummyContext(["NIFTY"])
    stub = _StubMessenger()
    monkeypatch.setattr(bot, "_ensure_messenger", lambda *_args, **_kwargs: stub)

    await bot.cmd_fresh(_DummyUpdate(chat), ctx)

    assert hub.calls == ["NIFTY"]
    message = stub.calls[-1]["text"]
    assert message.startswith("fresh")
    assert '"symbol":"NIFTY"' in message


@pytest.mark.asyncio
async def test_cmd_gate_reports_soft_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SoftRisk:
        def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
            return False, ("STALE",)

        def cooldown_remaining(self) -> float:
            return 0.0

    risk = _SoftRisk()
    deps = TelegramDeps(token="x", chat_id=1, app_version="test", risk_manager=risk)
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    ctx = _DummyContext()
    stub = _StubMessenger()
    monkeypatch.setattr(bot, "_ensure_messenger", lambda *_args, **_kwargs: stub)

    await bot.cmd_gate(_DummyUpdate(chat), ctx)

    assert stub.calls
    payload = stub.calls[-1]["text"]
    assert payload.startswith("gate")
    assert '"allowed":false' in payload
    assert '"soft_only":true' in payload


@pytest.mark.asyncio
async def test_cmd_guard_reports_soft_block(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SoftRisk:
        def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
            return False, ("STALE",)

        def cooldown_remaining(self) -> float:
            return 0.0

    class _OrderManager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []

        def _ensure_trading_allowed(
            self, *, symbol: str, side: str, quantity: int
        ) -> bool:
            self.calls.append((symbol, side, quantity))
            return False

    om = _OrderManager()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        order_manager=om,
        risk_manager=_SoftRisk(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    ctx = _DummyContext(["NIFTY"])
    stub = _StubMessenger()
    monkeypatch.setattr(bot, "_ensure_messenger", lambda *_args, **_kwargs: stub)

    await bot.cmd_guard(_DummyUpdate(chat), ctx)

    assert om.calls == [("NIFTY", "BUY", 1)]
    message = stub.calls[-1]["text"]
    assert message.startswith("guard")
    assert "soft_block" in message


@pytest.mark.asyncio
async def test_cmd_quote_prefers_data_hub() -> None:
    class _Hub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def get_quote(self, symbol: str, *, allow_pull: bool = True) -> dict[str, Any]:
            self.calls.append((symbol, allow_pull))
            return {
                "symbol": symbol,
                "ltp": 201.25,
                "bid": 201.0,
                "ask": 201.5,
                "timestamp": 1700000000.0,
            }

        def ingest_tick(
            self, tick: dict[str, Any]
        ) -> None:  # pragma: no cover - unused
            raise AssertionError("ingest_tick should not be called")

    class _FailingMDM:
        def pull_quote(self, symbol: str) -> dict[str, Any]:  # pragma: no cover
            raise AssertionError(
                "MDM pull_quote should not be used when hub returns data"
            )

    hub = _Hub()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        data_hub=hub,
        market_data_manager=_FailingMDM(),
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["nifty24feb18000ce"])

    await bot.cmd_quote(update, ctx)

    assert hub.calls == [("NIFTY24FEB18000CE", True)]
    message = chat.sent[-1]
    assert "NIFTY24FEB18000CE" in message
    assert "ltp=201.25" in message
    assert "bid=201.00" in message
    assert "ask=201.50" in message


@pytest.mark.asyncio
async def test_cmd_tail_rate_limits_and_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = TelegramDeps(token="x", chat_id=1, app_version="test")
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["root", "50"])

    tail_calls: list[tuple[str, int]] = []

    def _fake_tail(logger_name: str, lines: int) -> list[str]:
        tail_calls.append((logger_name, lines))
        return ["A" * 4000]

    monkeypatch.setattr(telegram_module.LOG_TAP, "tail", _fake_tail)
    monkeypatch.setenv("LOG_MAX_LINES_REPLY", "10")

    times = iter([100.0, 101.0])
    monkeypatch.setattr(telegram_module.time, "time", lambda: next(times))

    await bot.cmd_tail(update, ctx)

    assert tail_calls == [("root", 10)]
    message = chat.sent[-1]
    assert message.startswith("<pre>")
    assert message.endswith("</pre>")
    assert message.count("A") == 3400

    await bot.cmd_tail(update, ctx)

    assert len(tail_calls) == 1
    throttle_message = chat.sent[-1]
    assert "Tail throttled" in throttle_message


@pytest.mark.asyncio
async def test_cmd_test_flow_executes_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Hub:
        def __init__(self) -> None:
            self.quotes: list[tuple[str, bool]] = []
            self.ingested: list[dict[str, Any]] = []

        def get_quote(self, symbol: str, *, allow_pull: bool = True) -> dict[str, Any]:
            self.quotes.append((symbol, allow_pull))
            return {"symbol": symbol, "ltp": 150.0}

        def ingest_tick(self, tick: dict[str, Any]) -> None:
            self.ingested.append(dict(tick))

        def as_dict(self) -> dict[str, Any]:
            return {
                "orders": [
                    {
                        "order_id": "paper_1",
                        "symbol": "NIFTY24FEB18000CE",
                        "side": "BUY",
                        "quantity": 75,
                    }
                ],
                "positions": [
                    {
                        "symbol": "NIFTY24FEB18000CE",
                        "quantity": 75,
                        "side": "LONG",
                    }
                ],
            }

    class _SafeManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def place_order(self, **kwargs: Any) -> str:
            self.calls.append(dict(kwargs))
            return "test-order"

    class _Resolver:
        def lot_size_for_symbol(self, symbol: str) -> int:
            return 75

    hub = _Hub()
    safe_manager = _SafeManager()
    resolver = _Resolver()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        data_hub=hub,
        safe_order_manager=safe_manager,
        instrument_resolver=resolver,
        paper_mode_getters={"orders": lambda: True},
    )
    bot = TelegramBot(deps)
    chat = _DummyChat(1)
    update = types.SimpleNamespace(effective_chat=chat)
    ctx = _DummyContext(["nifty24feb18000ce"])

    loop = asyncio.get_running_loop()

    async def _immediate_run_in_executor(  # type: ignore[override]
        executor: Any,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(loop, "run_in_executor", _immediate_run_in_executor)

    await bot.cmd_test_flow(update, ctx)

    assert safe_manager.calls == [
        {
            "symbol": "NIFTY24FEB18000CE",
            "side": "BUY",
            "quantity": 75,
            "tag": "test_flow",
        }
    ]
    assert hub.quotes == [("NIFTY24FEB18000CE", True)]
    assert hub.ingested and hub.ingested[-1]["ltp"] == 200.0
    message = chat.sent[-1]
    assert "test-order" in message
    assert "positions" in message


def test_start_manual_polling_loop_blocks_duplicate_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = TelegramDeps(token='x', chat_id=1, app_version='test')
    first = TelegramBot(deps)
    second = TelegramBot(deps)

    created: list[tuple[str, Any]] = []

    class _Task:
        def done(self) -> bool:
            return False

    def _fake_create_task(coro: Any, *, name: str) -> _Task:
        created.append((name, coro))
        try:
            coro.close()
        except Exception:
            pass
        return _Task()

    monkeypatch.setattr(telegram_module.asyncio, 'create_task', _fake_create_task)
    TelegramBot._active_poller_instance = None

    first._start_manual_polling_loop()
    second._start_manual_polling_loop()

    assert len(created) == 1
    assert created[0][0] == 'telegram-manual-polling'


@pytest.mark.asyncio
async def test_polling_loop_recovers_from_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = TelegramDeps(token='x', chat_id=1, app_version='test')
    bot = TelegramBot(deps)
    bot._shutdown_event = asyncio.Event()

    class _FakeAppBot:
        def __init__(self, owner: TelegramBot) -> None:
            self._owner = owner
            self.calls = 0

        async def get_updates(self, **_: Any) -> list[Any]:
            self.calls += 1
            if self.calls == 1:
                raise Exception('Conflict: terminated by other getUpdates request')
            self._owner._shutdown_event.set()
            return []

    class _FakeApp:
        def __init__(self, owner: TelegramBot) -> None:
            self.bot = _FakeAppBot(owner)

        async def process_update(self, _update: Any) -> None:
            return None

    bot._app = _FakeApp(bot)

    async def _noop_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(telegram_module.asyncio, 'sleep', _noop_sleep)
    await bot._polling_loop()

    assert bot._polling_conflict_count == 1
    assert bot._metrics.polling_errors == 1
