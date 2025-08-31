from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo

from src.notifications.telegram_controller import TelegramController
from src.config import settings


class DummyRiskEngine:
    def __init__(self) -> None:
        self.state = SimpleNamespace(cooloff_until=None, cum_R_today=0.0)
        self.cfg = SimpleNamespace(max_daily_dd_R=5.0)

    def snapshot(self) -> dict:
        return {}


class FakeRunner:
    def __init__(self) -> None:
        self.within_window = True
        self.strategy_cfg = SimpleNamespace(
            min_bars_required=1,
            tz="Asia/Kolkata",
            atr_min=0.0,
            atr_max=5.0,
            score_trend_min=5,
            score_range_min=5,
            lower_score_temp=False,
            max_spread_pct_regular=1.0,
        )
        self.risk_engine = DummyRiskEngine()
        self.window_tuple = "-"
        self.now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        self.settings = SimpleNamespace(TZ="Asia/Kolkata")
        self.order_executor = SimpleNamespace(router_health=lambda: {}, api_health=lambda: {})
        self.data_source = SimpleNamespace(api_health=lambda: {}, get_last_bars=lambda n: None)
        self.last_plan = {}

    def telemetry_snapshot(self) -> dict:
        return {
            "signal": {
                "regime": "TREND",
                "score": 4,
                "atr_pct": 1.0,
                "micro": {},
                "reason_block": "no_option_token",
            },
            "bars": {
                "bar_count": 10,
                "last_bar_ts": self.now_ist.isoformat(),
            },
            "api_health": {
                "orders": {"state": "ok", "p95_ms": 1},
                "quote": {"state": "ok", "p95_ms": 1},
            },
            "router": {},
        }

    def runner_tick(self, dry: bool = False) -> dict:
        return {}


def test_audit_rootcause(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    runner = FakeRunner()
    tc = TelegramController(status_provider=lambda: {}, runner_tick=runner.runner_tick)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/audit"}})
    msg = sent[0]
    assert "reason_block: *no_option_token*" in msg
    assert "micro_spread: ❌ FAIL" in msg
    assert "micro_depth: ❌ FAIL" in msg
