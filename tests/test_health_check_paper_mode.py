from __future__ import annotations

import logging
from types import SimpleNamespace

from nifty_scalper_bot.core import app as app_module


class _Runner:
    def get_status(self) -> dict[str, object]:
        return {'running': False}


def test_health_check_paper_mode_not_live_disabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'PAPER')
    ctx = SimpleNamespace(
        strategy_runner=_Runner(),
        settings=SimpleNamespace(enable_live=False),
        trading_ready=False,
        live_orders_armed=False,
        shadow_mode_enabled=False,
        effective_mode='PAPER',
        readiness_mode='PAPER',
        started_mono=0.0,
    )
    caplog.set_level(logging.INFO)
    app_module._health_check(ctx)
    reasons = [r.reason for r in caplog.records]
    assert any('paper_runner_not_started' in msg for msg in reasons)
    assert not any('live_disabled' in msg for msg in reasons)
