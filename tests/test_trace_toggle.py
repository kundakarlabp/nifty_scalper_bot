from __future__ import annotations

import io
import logging
from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


class FakeRunner:
    def __init__(self) -> None:
        self.trace_ticks_remaining = 0
        self.last_plan = {
            "regime": "TREND",
            "score": 7,
            "atr_pct": 1.0,
            "micro": {"spread_pct": 0.1, "depth_ok": True},
            "rr": 2.0,
            "entry": 100,
            "sl": 95,
            "tp1": 105,
            "tp2": 110,
            "reason_block": None,
            "reasons": [],
        }
        self.logger = logging.getLogger("trace_test")
        self.log_stream = io.StringIO()
        handler = logging.StreamHandler(self.log_stream)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.order_executor = SimpleNamespace(router_health=lambda: {}, api_health=lambda: {})
        self.data_source = SimpleNamespace(api_health=lambda: {}, get_last_bars=lambda n: None)
        self.settings = SimpleNamespace(TZ="Asia/Kolkata")

    def runner_tick(self, dry: bool = False) -> dict:
        return {}

    def eval_step(self) -> None:
        if getattr(self, "trace_ticks_remaining", 0) > 0:
            p = self.last_plan or {}
            m = p.get("micro") or {}
            self.logger.info(
                "TRACE regime=%s score=%s atr%%=%.2f spread%%=%s depth=%s rr=%s entry=%s sl=%s tp1=%s tp2=%s block=%s reasons=%s",
                p.get("regime"),
                p.get("score"),
                float(p.get("atr_pct") or 0.0),
                m.get("spread_pct"),
                m.get("depth_ok"),
                p.get("rr"),
                p.get("entry"),
                p.get("sl"),
                p.get("tp1"),
                p.get("tp2"),
                p.get("reason_block"),
                p.get("reasons"),
            )
            self.trace_ticks_remaining -= 1


def test_trace_toggle(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    runner = FakeRunner()
    tc = TelegramController(status_provider=lambda: {}, runner_tick=runner.runner_tick)
    tc._send = lambda text, parse_mode=None: None
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/trace 3"}})
    assert runner.trace_ticks_remaining == 3
    for _ in range(5):
        runner.eval_step()
    assert runner.trace_ticks_remaining == 0
    logs = runner.log_stream.getvalue().splitlines()
    assert sum(1 for line in logs if line.startswith("TRACE")) == 3
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/traceoff"}})
    assert runner.trace_ticks_remaining == 0
