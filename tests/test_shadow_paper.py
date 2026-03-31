from __future__ import annotations

from nifty_scalper_bot.config.settings import ShadowSettings
from nifty_scalper_bot.shadow.shadow_paper import ShadowPaperTrader


def test_shadow_trader_triggers_disable_on_drift() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    reasons: list[str] = []

    def notifier(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    def disable(reason: str) -> None:
        reasons.append(reason)

    trader = ShadowPaperTrader(
        settings=ShadowSettings(
            drift_threshold_pct=1.0,
            auto_disable_live_on_drift=True,
            drift_sustain_seconds=0.0,
            alert_threshold_pct=0.5,
            alert_trend_delta_pct=0.1,
            alert_cooldown_seconds=0.0,
        ),
        live_equity_fn=lambda: 100.0,
        notifier=notifier,
        disable_live_callback=disable,
    )
    trader.record_order("NIFTY", "BUY", 10, 20.0)
    assert {name for name, _ in events} >= {"shadow_drift", "shadow_drift_disable"}
    assert reasons == ["shadow-drift"]
