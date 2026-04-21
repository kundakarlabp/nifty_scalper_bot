from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core import app as app_module
from nifty_scalper_bot.utils.market_hours import MarketState


class _Runner:
    def __init__(self, status: dict[str, object]) -> None:
        self._status = status

    def get_status(self) -> dict[str, object]:
        return dict(self._status)


def test_health_check_logs_reason_aware_runner_inactive(
    monkeypatch,
    caplog,
) -> None:
    ctx = SimpleNamespace(
        strategy_runner=_Runner({"running": False}),
        settings=SimpleNamespace(enable_live=True),
        shadow_mode_enabled=False,
    )
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)

    caplog.set_level("INFO")
    app_module._health_check(ctx)

    assert any("market_closed" in rec.message for rec in caplog.records)

